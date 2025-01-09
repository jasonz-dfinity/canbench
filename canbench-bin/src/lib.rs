//! A module for running benchmarks.
mod print_benchmark;
mod results_file;
use canbench_rs::BenchResult;
use candid::Principal;
use flate2::read::GzDecoder;
use pocket_ic::common::rest::BlobCompression;
use pocket_ic::{PocketIc, PocketIcBuilder, WasmResult};
use print_benchmark::print_benchmark;
use results_file::VersionError;
use std::{collections::BTreeMap, env, fs::File, io::Read, path::PathBuf, process::Command};
use wasmparser::Parser as WasmParser;

// The prefix benchmarks are expected to have in their name.
// Other queries exposed by the canister are ignored.
const BENCH_PREFIX: &str = "__canbench__";

const POCKET_IC_LINUX_SHA: &str =
    "95e3bb14977228efbb5173ea3e044e6b6c8420bb1b3342fa530e3c11f3e9f0cd";
const POCKET_IC_MAC_SHA: &str = "87582439bf456221256c66e86b382a56f5df7a6a8da85738eaa233d2ada3ed47";

/// Runs the benchmarks on the canister available in the provided `canister_wasm_path`.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmarks(
    canister_wasm_path: &PathBuf,
    pattern: Option<String>,
    init_args: Vec<u8>,
    persist: bool,
    results_file: &PathBuf,
    verbose: bool,
    integrity_check: bool,
    tracing: bool,
    runtime_path: &PathBuf,
    stable_memory_path: Option<PathBuf>,
) {
    maybe_download_pocket_ic(runtime_path, verbose, integrity_check);

    let current_results = match results_file::read(results_file) {
        Ok(current_results) => current_results,
        Err(VersionError {
            our_version,
            their_version,
        }) => {
            eprintln!("canbench is at version {our_version} while the results were generated with version {their_version}. Please upgrade canbench.");
            std::process::exit(1);
        }
    };

    let mut benchmark_wasm = read_wasm(canister_wasm_path);

    // Extract the benchmark functions in the Wasm.
    // TODO: Use walrus for this.
    let benchmark_fns = extract_benchmark_fns(&benchmark_wasm);
    let names_mapping = instrumentation::extract_function_names(&benchmark_wasm);

    let tracing_wasm = if tracing {
        Some(instrumentation::instrument_wasm_for_tracing(
            &mut benchmark_wasm,
        ))
    } else {
        None
    };

    // TODO: remove this, since it's for development purposes only.
    if let Some(tracing_wasm) = &tracing_wasm {
        std::fs::write(
            canister_wasm_path.with_file_name("instrumented.wasm"),
            tracing_wasm,
        )
        .expect("Failed to write instrumented wasm");
    }

    // Initialize PocketIC
    let (pocket_ic, benchmark_canister_id, tracing_canister_id) = init_pocket_ic(
        runtime_path,
        benchmark_wasm,
        tracing_wasm,
        stable_memory_path,
        init_args,
    );

    // Run the benchmarks
    let mut results = BTreeMap::new();
    let mut num_executed_bench_fns = 0;
    for bench_fn in &benchmark_fns {
        if let Some(pattern) = &pattern {
            if !bench_fn.contains(pattern) {
                continue;
            }
        }

        println!();
        println!("---------------------------------------------------");
        println!();

        let result = run_benchmark(&pocket_ic, benchmark_canister_id, bench_fn);
        print_benchmark(bench_fn, &result, current_results.get(bench_fn));

        if let Some(tracing_canister_id) = tracing_canister_id {
            run_benchmark_with_tracing(
                &pocket_ic,
                tracing_canister_id,
                bench_fn,
                &names_mapping,
                results_file,
                result.total.instructions,
            );
        }

        results.insert(bench_fn.to_string(), result);
        num_executed_bench_fns += 1;
    }

    println!();
    println!("---------------------------------------------------");

    if verbose {
        println!();
        println!(
            "Executed {num_executed_bench_fns} of {} benchmarks.",
            benchmark_fns.len()
        );
    }

    // Persist the result if requested.
    if persist {
        results_file::write(results_file, results);
        println!(
            "Successfully persisted results to {}",
            results_file.display()
        );
    }
}

// Downloads PocketIC if it's not already downloaded.
fn maybe_download_pocket_ic(path: &PathBuf, verbose: bool, integrity_check: bool) {
    match (path.exists(), integrity_check) {
        (true, true) => {
            // Verify that it's the version we expect it to be.

            let pocket_ic_sha = sha256::try_digest(path).unwrap();
            let expected_sha = expected_runtime_digest();

            if pocket_ic_sha != expected_sha {
                eprintln!(
                    "Runtime has incorrect digest. Expected: {}, actual: {}",
                    expected_sha, pocket_ic_sha
                );
                eprintln!("Runtime will be redownloaded...");
                download_pocket_ic(path, verbose);
            }
        }
        (true, false) => {} // Nothing to do
        (false, _) => {
            // Pocket IC not present. Download it.
            download_pocket_ic(path, verbose);
        }
    }
}

fn download_pocket_ic(path: &PathBuf, verbose: bool) {
    const POCKET_IC_URL_PREFIX: &str =
        "https://github.com/dfinity/pocketic/releases/download/7.0.0/pocket-ic-x86_64-";
    if verbose {
        println!("Downloading runtime (will be cached for future uses)...");
    }

    // Create the canbench directory if it doesn't exist.
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let os = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        panic!("Unsupported operating system");
    };

    let url = format!("{}{}.gz", POCKET_IC_URL_PREFIX, os);
    let pocket_ic_compressed = reqwest::blocking::get(url)
        .unwrap()
        .bytes()
        .expect("Failed to download PocketIC");

    let mut decoder = GzDecoder::new(&pocket_ic_compressed[..]);
    let mut file = File::create(path).expect("Failed to create PocketIC file");

    std::io::copy(&mut decoder, &mut file).expect("Failed to write PocketIC file");
    // Make the file executable.
    Command::new("chmod").arg("+x").arg(path).status().unwrap();
}

// Runs the given benchmark.
fn run_benchmark(pocket_ic: &PocketIc, canister_id: Principal, bench_fn: &str) -> BenchResult {
    match pocket_ic.query_call(
        canister_id,
        Principal::anonymous(),
        &format!("{}{}", BENCH_PREFIX, bench_fn),
        b"DIDL\x00\x00".to_vec(),
    ) {
        Ok(wasm_res) => match wasm_res {
            WasmResult::Reply(res) => {
                let res: BenchResult =
                    candid::decode_one(&res).expect("error decoding benchmark result");
                res
            }
            WasmResult::Reject(output_str) => {
                eprintln!(
                    "Error executing benchmark {}. Error:\n{}",
                    bench_fn, output_str
                );
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error executing benchmark {}. Error:\n{}", bench_fn, e);
            std::process::exit(1);
        }
    }
}

fn run_benchmark_with_tracing(
    pocket_ic: &PocketIc,
    canister_id: Principal,
    bench_fn: &str,
    names_mapping: &BTreeMap<i32, String>,
    results_file: &PathBuf,
    bench_instructions: u64,
) {
    let traces: Result<Vec<(i32, i64)>, String> = match pocket_ic.query_call(
        canister_id,
        Principal::anonymous(),
        &format!("__tracing__{}", bench_fn),
        b"DIDL\x00\x00".to_vec(),
    ) {
        Ok(wasm_res) => match wasm_res {
            WasmResult::Reply(res) => {
                let res: Result<Vec<(i32, i64)>, String> =
                    candid::decode_one(&res).expect("error decoding tracing result");
                res
            }
            WasmResult::Reject(output_str) => {
                eprintln!(
                    "Error tracing benchmark {}. Error:\n{}",
                    bench_fn, output_str
                );
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error tracing benchmark {}. Error:\n{}", bench_fn, e);
            std::process::exit(1);
        }
    };
    match traces {
        Ok(traces) => instrumentation::write_tracing_to_file(
            traces,
            bench_instructions,
            names_mapping,
            bench_fn,
            results_file.with_file_name(format!("{}.svg", bench_fn)),
        )
        .expect("failed to write tracing results"),
        Err(e) => {
            eprint!("Error tracing benchmark {}. Error:\n{}", bench_fn, e);
        }
    }
}

fn read_wasm(canister_wasm_path: &PathBuf) -> Vec<u8> {
    // Parse the canister's wasm.
    let wasm = std::fs::read(canister_wasm_path).unwrap_or_else(|_| {
        eprintln!(
            "Couldn't read file at {}. Are you sure the file exists?",
            canister_wasm_path.display()
        );
        std::process::exit(1);
    });

    // Decompress the wasm if it's gzipped.
    match canister_wasm_path.extension().unwrap().to_str() {
        Some("gz") => {
            // Decompress the wasm if it's gzipped.
            let mut decoder = GzDecoder::new(&wasm[..]);
            let mut decompressed_wasm = vec![];
            decoder.read_to_end(&mut decompressed_wasm).unwrap();
            decompressed_wasm
        }
        _ => wasm,
    }
}

mod instrumentation {
    use super::*;

    use walrus::ir::*;
    use walrus::*;

    fn make_trace_func(
        module: &mut Module,
        trace_start_address: GlobalId,
        performance_counter: FunctionId,
    ) -> FunctionId {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let mut body = builder.func_body();
        // TODO: is it safe to assume there is only one memory?
        let memory = module.get_memory_id().unwrap();

        let func_id = module.locals.add(ValType::I32);
        let num_logs_address = module.locals.add(ValType::I32);
        let num_logs_before = module.locals.add(ValType::I64);
        let new_log_address = module.locals.add(ValType::I32);
        let store_kind_i32 = StoreKind::I32 { atomic: false };
        let load_kind_i32 = LoadKind::I32 { atomic: false };
        let store_kind_i64 = StoreKind::I64 { atomic: false };
        let load_kind_i64 = LoadKind::I64 { atomic: false };
        let mem_arg_i32 = MemArg {
            offset: 0,
            align: 4,
        };
        let mem_arg_i64 = MemArg {
            offset: 0,
            align: 8,
        };

        // Check whether tracing is enabled, leaving the value (0 or 1 as i32) in the stack.
        let is_tracing_enabled = |body: &mut InstrSeqBuilder| {
            body.global_get(trace_start_address)
                .load(memory, load_kind_i32, mem_arg_i32)
                .i32_const(1)
                .binop(BinaryOp::I32Eq);
        };
        // Increment the number of logs by 1, while setting `num_logs_before` to the previous value,
        // and `num_logs_address` to the address of the number of logs.
        let increment_num_logs = |body: &mut InstrSeqBuilder| {
            body.global_get(trace_start_address)
                .i32_const(4)
                .binop(BinaryOp::I32Add)
                .local_tee(num_logs_address)
                .local_get(num_logs_address)
                .load(memory, load_kind_i64, mem_arg_i64)
                .local_tee(num_logs_before)
                .i64_const(1)
                .binop(BinaryOp::I64Add)
                .store(memory, store_kind_i64, mem_arg_i64);
        };
        // Assuming the number of logs is less than 100_000 (therefore the number can be wrapped as
        // i32), write a log entry.
        let write_log = |body: &mut InstrSeqBuilder| {
            increment_num_logs(body);
            body.local_get(num_logs_before)
                .unop(UnaryOp::I32WrapI64)
                .i32_const(12)
                .binop(BinaryOp::I32Mul)
                .i32_const(12) // 4 bytes for enabled flag, 4 bytes for number of entries
                .binop(BinaryOp::I32Add)
                .global_get(trace_start_address)
                .binop(BinaryOp::I32Add)
                .local_tee(new_log_address)
                .local_get(func_id)
                .store(memory, store_kind_i32, mem_arg_i32)
                .local_get(new_log_address)
                .i32_const(4)
                .binop(BinaryOp::I32Add)
                .i32_const(0)
                .call(performance_counter)
                .store(memory, store_kind_i64, mem_arg_i64);
        };
        let write_log_if_not_full = |body: &mut InstrSeqBuilder| {
            body.global_get(trace_start_address)
                .i32_const(4)
                .binop(BinaryOp::I32Add)
                .load(memory, load_kind_i64, mem_arg_i64)
                .i64_const(100_000)
                .binop(BinaryOp::I64LtU)
                .if_else(None, write_log, increment_num_logs);
        };

        is_tracing_enabled(&mut body);
        body.if_else(None, write_log_if_not_full, |_| {});
        builder.finish(vec![func_id], &mut module.funcs)
    }

    fn inject_prepare_tracing_call(
        types: &ModuleTypes,
        traces_start_address: GlobalId,
        prepare_func: FunctionId,
        func: &mut LocalFunction,
    ) {
        // Put the original function body inside a block, so that if the code
        // use br_if/br_table to exit the function, we can still output the exit signal.
        let start_id = func.entry_block();
        let original_block = func.block_mut(start_id);
        let start_instrs = original_block.instrs.split_off(0);
        let start_ty = match original_block.ty {
            InstrSeqType::MultiValue(id) => {
                let valtypes = types.results(id);
                InstrSeqType::Simple(match valtypes.len() {
                    0 => None,
                    1 => Some(valtypes[0]),
                    _ => unreachable!("Multivalue return not supported"),
                })
            }
            // top-level block is using the function signature
            InstrSeqType::Simple(_) => unreachable!(),
        };
        let mut inner_start = func.builder_mut().dangling_instr_seq(start_ty);
        *(inner_start.instrs_mut()) = start_instrs;
        let inner_start_id = inner_start.id();
        let mut start_builder = func.builder_mut().func_body();
        start_builder
            .call(prepare_func)
            .global_set(traces_start_address)
            .instr(Block {
                seq: inner_start_id,
            });
        let mut stack = vec![inner_start_id];
        while let Some(seq_id) = stack.pop() {
            let mut builder = func.builder_mut().instr_seq(seq_id);
            let original = builder.instrs_mut();
            let mut instrs = vec![];
            for (instr, loc) in original.iter() {
                match instr {
                    Instr::Block(Block { seq }) | Instr::Loop(Loop { seq }) => {
                        stack.push(*seq);
                        instrs.push((instr.clone(), *loc));
                    }
                    Instr::IfElse(IfElse {
                        consequent,
                        alternative,
                    }) => {
                        stack.push(*alternative);
                        stack.push(*consequent);
                        instrs.push((instr.clone(), *loc));
                    }
                    Instr::Return(_) => {
                        instrs.push((
                            Instr::Br(Br {
                                block: inner_start_id,
                            }),
                            *loc,
                        ));
                    }
                    // redirect br,br_if,br_table to inner seq id
                    Instr::Br(Br { block }) if *block == start_id => {
                        instrs.push((
                            Instr::Br(Br {
                                block: inner_start_id,
                            }),
                            *loc,
                        ));
                    }
                    Instr::BrIf(BrIf { block }) if *block == start_id => {
                        instrs.push((
                            Instr::BrIf(BrIf {
                                block: inner_start_id,
                            }),
                            *loc,
                        ));
                    }
                    Instr::BrTable(BrTable { blocks, default }) => {
                        let mut blocks = blocks.clone();
                        for i in 0..blocks.len() {
                            if let Some(id) = blocks.get_mut(i) {
                                if *id == start_id {
                                    *id = inner_start_id
                                };
                            }
                        }
                        let default = if *default == start_id {
                            inner_start_id
                        } else {
                            *default
                        };
                        instrs.push((Instr::BrTable(BrTable { blocks, default }), *loc));
                    }
                    _ => instrs.push((instr.clone(), *loc)),
                }
            }
            *original = instrs;
        }
    }

    fn inject_tracing(
        types: &ModuleTypes,
        trace_func: FunctionId,
        id: FunctionId,
        func: &mut LocalFunction,
    ) {
        // Put the original function body inside a block, so that if the code
        // use br_if/br_table to exit the function, we can still output the exit signal.
        let start_id = func.entry_block();
        let original_block = func.block_mut(start_id);
        let start_instrs = original_block.instrs.split_off(0);
        let start_ty = match original_block.ty {
            InstrSeqType::MultiValue(id) => {
                let valtypes = types.results(id);
                InstrSeqType::Simple(match valtypes.len() {
                    0 => None,
                    1 => Some(valtypes[0]),
                    _ => unreachable!("Multivalue return not supported"),
                })
            }
            // top-level block is using the function signature
            InstrSeqType::Simple(_) => unreachable!(),
        };
        let mut inner_start = func.builder_mut().dangling_instr_seq(start_ty);
        *(inner_start.instrs_mut()) = start_instrs;
        let inner_start_id = inner_start.id();
        let mut start_builder = func.builder_mut().func_body();
        start_builder
            .i32_const(id.index() as i32)
            .call(trace_func)
            .instr(Block {
                seq: inner_start_id,
            })
            // Since the func index can be 0, we subtract 1 here. Mapping: 0->-1, 1->-2, ...,
            // i32::MAX->i32::MIN.
            .i32_const(-(id.index() as i32) - 1)
            .call(trace_func);
        let mut stack = vec![inner_start_id];
        while let Some(seq_id) = stack.pop() {
            let mut builder = func.builder_mut().instr_seq(seq_id);
            let original = builder.instrs_mut();
            let mut instrs = vec![];
            for (instr, loc) in original.iter() {
                match instr {
                    Instr::Block(Block { seq }) | Instr::Loop(Loop { seq }) => {
                        stack.push(*seq);
                        instrs.push((instr.clone(), *loc));
                    }
                    Instr::IfElse(IfElse {
                        consequent,
                        alternative,
                    }) => {
                        stack.push(*alternative);
                        stack.push(*consequent);
                        instrs.push((instr.clone(), *loc));
                    }
                    Instr::Return(_) => {
                        instrs.push((
                            Instr::Br(Br {
                                block: inner_start_id,
                            }),
                            *loc,
                        ));
                    }
                    // redirect br,br_if,br_table to inner seq id
                    Instr::Br(Br { block }) if *block == start_id => {
                        instrs.push((
                            Instr::Br(Br {
                                block: inner_start_id,
                            }),
                            *loc,
                        ));
                    }
                    Instr::BrIf(BrIf { block }) if *block == start_id => {
                        instrs.push((
                            Instr::BrIf(BrIf {
                                block: inner_start_id,
                            }),
                            *loc,
                        ));
                    }
                    Instr::BrTable(BrTable { blocks, default }) => {
                        let mut blocks = blocks.clone();
                        for i in 0..blocks.len() {
                            if let Some(id) = blocks.get_mut(i) {
                                if *id == start_id {
                                    *id = inner_start_id
                                };
                            }
                        }
                        let default = if *default == start_id {
                            inner_start_id
                        } else {
                            *default
                        };
                        instrs.push((Instr::BrTable(BrTable { blocks, default }), *loc));
                    }
                    _ => instrs.push((instr.clone(), *loc)),
                }
            }
            *original = instrs;
        }
    }

    fn adjust_traces_for_overhead(
        traces: Vec<(i32, i64)>,
        bench_instructions: u64,
    ) -> Vec<(i32, i64)> {
        // TODO: make the output non-decreassing, and adjust the last entry.
        let num_logs = traces.len() - 2;
        let overhead = (traces[num_logs].1 as f64 - bench_instructions as f64) / (num_logs as f64);
        traces
            .into_iter()
            .enumerate()
            .map(|(i, (id, count))| {
                if i <= num_logs {
                    (id, count - (overhead * i as f64) as i64)
                } else {
                    (id, count - (overhead * num_logs as f64) as i64)
                }
            })
            .collect()
    }

    /// Renders the tracing to a file. Adapted from
    /// https://github.com/dfinity/ic-repl/blob/master/src/tracing.rs
    pub(super) fn write_tracing_to_file(
        input: Vec<(i32, i64)>,
        bench_instructions: u64,
        names: &BTreeMap<i32, String>,
        bench_fn: &str,
        filename: PathBuf,
    ) -> Result<(), String> {
        // TODO: make an aggregated graph in addition to time-based graph.
        let input = adjust_traces_for_overhead(input, bench_instructions);
        use inferno::flamegraph::{from_reader, Options};
        let mut stack = Vec::new();
        let mut prefix = Vec::new();
        let mut result = Vec::new();
        let mut prev = None;
        for (id, count) in input.into_iter() {
            if id >= 0 {
                stack.push((id, count, 0));
                let name = if id < i32::MAX {
                    match names.get(&id) {
                        Some(name) => name.clone(),
                        None => "func_".to_string() + &id.to_string(),
                    }
                } else {
                    bench_fn.to_string()
                };
                prefix.push(name);
            } else {
                // Negative id means the end of a function. Mapping: -1->0, -2->1, ...,
                // i32::MIN->i32::MAX.
                let end_id = -(id + 1);
                match stack.pop() {
                    None => return Err("pop empty stack".to_string()),
                    Some((start_id, start, children)) => {
                        if start_id != end_id {
                            return Err("func id mismatch".to_string());
                        }
                        let cost = count - start;
                        let frame = prefix.join(";");
                        prefix.pop().unwrap();
                        if let Some((parent, parent_cost, children_cost)) = stack.pop() {
                            stack.push((parent, parent_cost, children_cost + cost));
                        }
                        match prev {
                            Some(prev) if prev == frame => {
                                // Add an empty spacer to avoid collapsing adjacent same-named calls
                                // See https://github.com/jonhoo/inferno/issues/185#issuecomment-671393504
                                result.push(format!("{};spacer 0", prefix.join(";")));
                            }
                            _ => (),
                        }
                        result.push(format!("{} {}", frame, cost - children));
                        prev = Some(frame);
                    }
                }
            }
        }
        let is_trace_incomplete = !stack.is_empty();
        let mut opt = Options::default();
        opt.count_name = "instructions".to_string();
        let bench_fn = if is_trace_incomplete {
            bench_fn.to_string() + " (incomplete)"
        } else {
            bench_fn.to_string()
        };
        opt.title = bench_fn;
        opt.image_width = Some(1024);
        opt.flame_chart = true;
        opt.no_sort = true;
        // Reserve result order to make flamegraph from left to right.
        // See https://github.com/jonhoo/inferno/issues/236
        result.reverse();
        let logs = result.join("\n");
        let reader = std::io::Cursor::new(logs);
        let mut writer = std::fs::File::create(&filename).map_err(|e| e.to_string())?;
        from_reader(&mut opt, reader, &mut writer).map_err(|e| e.to_string())?;
        println!("Flamegraph written to {}", filename.display());
        Ok(())
    }

    pub(super) fn extract_function_names(wasm: &Vec<u8>) -> BTreeMap<i32, String> {
        let mut config = ModuleConfig::new();
        config.generate_producers_section(false);
        let module = config.parse(wasm).expect("failed to parse wasm");
        module
            .funcs
            .iter()
            .filter_map(|f| {
                if matches!(f.kind, FunctionKind::Local(_)) {
                    use rustc_demangle::demangle;
                    let name = f.name.as_ref()?;
                    let demangled = format!("{:#}", demangle(name));
                    Some((f.id().index() as i32, demangled))
                } else {
                    None
                }
            })
            .collect()
    }

    pub(super) fn instrument_wasm_for_tracing(wasm: &Vec<u8>) -> Vec<u8> {
        let mut config = ModuleConfig::new();
        // TODO: why is this needed?
        config.generate_producers_section(false);

        let mut module = config.parse(wasm).expect("failed to parse wasm");
        let performance_counter = module
            .imports
            .get_func("ic0", "performance_counter")
            .unwrap();
        let traces_start_address =
            module
                .globals
                .add_local(ValType::I32, true, false, ConstExpr::Value(Value::I32(0)));
        let trace_func = make_trace_func(&mut module, traces_start_address, performance_counter);
        for (id, func) in module.funcs.iter_local_mut() {
            if id != trace_func {
                inject_tracing(&module.types, trace_func, id, func);
            }
        }

        let prepare_func = module.funcs.by_name("__prepare_tracing").unwrap();
        let bench_funcs: Vec<_> = module
            .funcs
            .iter()
            .filter_map(|f| {
                if f.name
                    .as_ref()
                    .map_or(false, |name| name.starts_with("canister_query __tracing__"))
                {
                    Some(f.id())
                } else {
                    None
                }
            })
            .collect();
        for (id, func) in module.funcs.iter_local_mut() {
            if bench_funcs.contains(&id) {
                inject_prepare_tracing_call(
                    &module.types,
                    traces_start_address,
                    prepare_func,
                    func,
                );
            }
        }

        for (id, section) in module.customs.iter() {
            println!("Custom section: {:?} {:?}", id, section.name());
        }

        module.emit_wasm()
    }
}

// Extract the benchmarks that need to be run.
fn extract_benchmark_fns(wasm: &Vec<u8>) -> Vec<String> {
    let prefix = format!("canister_query {BENCH_PREFIX}");

    WasmParser::new(0)
        .parse_all(&wasm)
        .filter_map(|section| match section {
            Ok(wasmparser::Payload::ExportSection(export_section)) => {
                let queries: Vec<_> = export_section
                    .into_iter()
                    .filter_map(|export| {
                        if let Ok(export) = export {
                            if export.name.starts_with(&prefix) {
                                return Some(
                                    export
                                        .name
                                        .split(&prefix)
                                        .last()
                                        .expect("query must have a name.")
                                        .to_string(),
                                );
                            }
                        }

                        None
                    })
                    .collect();

                Some(queries)
            }
            _ => None,
        })
        .flatten()
        .collect()
}

// Sets the environment variable to the target value if it's not already set.
fn set_env_var_if_unset(key: &str, target_value: &str) {
    if std::env::var(key).is_err() {
        std::env::set_var(key, target_value);
    }
}

// Initializes PocketIC and installs the canister to benchmark.
fn init_pocket_ic(
    path: &PathBuf,
    benchmark_wasm: Vec<u8>,
    tracing_wasm: Option<Vec<u8>>,
    stable_memory_path: Option<PathBuf>,
    init_args: Vec<u8>,
) -> (PocketIc, Principal, Option<Principal>) {
    // PocketIC is used for running the benchmark.
    // Set the appropriate ENV variables
    std::env::set_var("POCKET_IC_BIN", path);
    set_env_var_if_unset("POCKET_IC_MUTE_SERVER", "1");
    let pocket_ic = PocketIcBuilder::new()
        .with_max_request_time_ms(None)
        .with_benchmarking_application_subnet()
        .build();

    let stable_memory = stable_memory_path.map(|path| match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("Error reading stable memory file {}", path.display());
            eprintln!("Error: {}", err);
            std::process::exit(1);
        }
    });

    let tracing_canister_id = tracing_wasm
        .map(|wasm| init_canister(&pocket_ic, wasm, init_args.clone(), stable_memory.clone()));
    let benchmark_canister_id = init_canister(&pocket_ic, benchmark_wasm, init_args, stable_memory);

    (pocket_ic, benchmark_canister_id, tracing_canister_id)
}

fn init_canister(
    pocket_ic: &PocketIc,
    wasm: Vec<u8>,
    init_args: Vec<u8>,
    stable_memory: Option<Vec<u8>>,
) -> Principal {
    let canister_id = pocket_ic.create_canister();
    pocket_ic.add_cycles(canister_id, 1_000_000_000_000_000);
    pocket_ic.install_canister(canister_id, wasm, init_args, None);
    // Load the canister's stable memory if stable memory is specified.
    if let Some(stable_memory) = stable_memory {
        pocket_ic.set_stable_memory(canister_id, stable_memory, BlobCompression::NoCompression);
    }
    canister_id
}

// Public only for tests.
#[doc(hidden)]
pub fn expected_runtime_digest() -> &'static str {
    match env::consts::OS {
        "linux" => POCKET_IC_LINUX_SHA,
        "macos" => POCKET_IC_MAC_SHA,
        _ => panic!("only linux and macos are currently supported."),
    }
}

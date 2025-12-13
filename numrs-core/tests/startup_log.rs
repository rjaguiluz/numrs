use numrs::print_startup_log;

#[test]
fn startup_log_runs() {
    // Call the startup logger to ensure it compiles and runs. No assertion needed — it's a smoke test.
    print_startup_log();
}

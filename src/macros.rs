macro_rules! undefined {
    () => (panic!("quantity undefined"));
    ($($arg:tt)+) => (panic!("quantity undefined: {}", std::format_args!($($arg)+)));
}

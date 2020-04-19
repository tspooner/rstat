# rstat

[![Crates.io](https://img.shields.io/crates/v/rstat.svg)](https://crates.io/crates/rstat)
[![Build Status](https://travis-ci.org/tspooner/rstat.svg?branch=master)](https://travis-ci.org/tspooner/rstat)
[![Coverage Status](https://coveralls.io/repos/github/tspooner/rstat/badge.svg?branch=master)](https://coveralls.io/github/tspooner/rstat?branch=master)


> Probability distributions and _statistics_ in Rust with
> integrated _fitting_ routines, _convolution_ support and _mixtures_.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
rstat = "0.4"
```

### Feature `serde`
`rstat` support serialisation via the `serde` feature. This activates both
`serde` itself, `ndarray/serde` and `spaces/serialize`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to the angularjs
commit message conventions (see
[here](https://gist.github.com/stephenparish/9941e89d80e2bc58a153)).

## License
[MIT](https://choosealicense.com/licenses/mit/)

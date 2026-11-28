# Changelog

## [3.0.0](https://github.com/CoreySpohn/zodi/compare/v2.0.0...v3.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **zodi:** close zodiacal and exozodi reads at their working precision
* remove zodi.jax and the jax extra, the equinox models now live in skyscapes

### Features

* remove zodi.jax and the jax extra, the equinox models now live in skyscapes ([3560dad](https://github.com/CoreySpohn/zodi/commit/3560dad702fa2ffb1546c94243cdf00dbcfd505a))


### Bug Fixes

* **zodi:** close zodiacal and exozodi reads at their working precision ([64499b6](https://github.com/CoreySpohn/zodi/commit/64499b6bfc1aac4069a2fdbca3e3ed0fff640d99))
* **zodi:** decide the grey-scatter cutoff on the host wavelength before narrowing ([e603d5b](https://github.com/CoreySpohn/zodi/commit/e603d5b3eedabb09a3870634d2efe93a59c073c1))

## [2.0.0](https://github.com/CoreySpohn/zodi/compare/v1.1.0...v2.0.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* declare zodiacal and exozodiacal spectral breaks through the hwoutils protocol
* delete the zodi.jax Leinert wrappers; models and callers use the core functions
* zodi owns its Equinox local-zodi models under zodi.jax and drops the unused band rules

### Features

* declare zodiacal and exozodiacal spectral breaks through the hwoutils protocol ([66a829b](https://github.com/CoreySpohn/zodi/commit/66a829b852fb106a7900d6900471f8af0098442b))
* typed dust amplitude records for the three meanings of one zodi ([6de44a8](https://github.com/CoreySpohn/zodi/commit/6de44a8e7fc4f9aa5729662dc33b9dbebc55bee4))
* zodi owns its Equinox local-zodi models under zodi.jax and drops the unused band rules ([906b651](https://github.com/CoreySpohn/zodi/commit/906b651342257bbbeb8f2827cf3720185dce066f))
* zodi.jax owns the analytic AYO exozodi model ([a4151af](https://github.com/CoreySpohn/zodi/commit/a4151afcc7d70a2086e3d450d55edbf832c93077))


### Code Refactoring

* delete the zodi.jax Leinert wrappers; models and callers use the core functions ([47a979e](https://github.com/CoreySpohn/zodi/commit/47a979e55d6be09e6a4b89f8d5385fb1271a7294))

## [1.1.0](https://github.com/CoreySpohn/zodi/compare/v1.0.0...v1.1.0) (2026-11-12)


### Features

* **zodi:** AYO default zodi color in AB, with the F_nu conversion ([afce5db](https://github.com/CoreySpohn/zodi/commit/afce5db3a8f955a85833236ed1f6e9c5a3e27cb6))

## [1.0.0](https://github.com/CoreySpohn/zodi/compare/v0.1.0...v1.0.0) (2026-09-16)


### ⚠ BREAKING CHANGES

* measure exozodi radial scaling from the EEID to remove the luminosity double count

### Features

* tag tests with ineedvalidation evidence, pin the clamp contract, and guard um wavelength arguments against nm values ([50de1ce](https://github.com/CoreySpohn/zodi/commit/50de1cebc5f8029cbfab76c601efcdc0491f801f))


### Bug Fixes

* measure exozodi radial scaling from the EEID to remove the luminosity double count ([ddc2c07](https://github.com/CoreySpohn/zodi/commit/ddc2c07bdb18c5ea58fc80ce2f3fe5c1de67a223))

## 0.1.0 (2026-07-18)


### Features

* scaffold the library with array API dispatch, clamped interpolation, and a backend-parity suite ([b2cd5e7](https://github.com/CoreySpohn/zodi/commit/b2cd5e76a46904d931177a02ab290adbf8254231))
* v0.1 core (Leinert zodi, exozodi chain + grey-scatterer flambda, units arbiter, cross-code validation, docs) ([31ffd72](https://github.com/CoreySpohn/zodi/commit/31ffd72039d12c04654f8f8c6dc0465faa421198))
* wire release automation -- release-please, PyPI trusted publishing, hatch-vcs tag-driven versioning ([1b39f2c](https://github.com/CoreySpohn/zodi/commit/1b39f2cf3f3a5cc1eb3f0842f328019ecb36acf3))

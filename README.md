# surp

Information theory primitives: entropy, KL divergence, mutual information.
Useful for evaluation and discrete distribution comparisons.

Dual-licensed under MIT or Apache-2.0.

```rust
use surp::{entropy, kl_divergence};

let p = [0.25, 0.25, 0.25, 0.25];
let q = [0.50, 0.25, 0.125, 0.125];

let h = entropy(&p);
let kl = kl_divergence(&p, &q);

println!("H(p)={h}, KL(p||q)={kl}");
```

For more, see the crate docs and `docs/`.
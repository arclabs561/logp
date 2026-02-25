//! Measure similarity between bag-of-words distributions using information theory.
//!
//! Constructs simple word frequency distributions from two short texts,
//! then compares them using JS divergence and mutual information.
//!
//! Run: cargo run --example text_similarity

use logp::{entropy_nats, jensen_shannon_divergence, kl_divergence};
use std::collections::HashMap;

fn word_distribution(text: &str) -> (Vec<String>, Vec<f64>) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in text.split_whitespace() {
        let w = word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
        if !w.is_empty() {
            *counts.entry(w).or_default() += 1;
        }
    }
    let total: usize = counts.values().sum();
    let mut words: Vec<String> = counts.keys().cloned().collect();
    words.sort();
    let probs: Vec<f64> = words.iter().map(|w| counts[w] as f64 / total as f64).collect();
    (words, probs)
}

fn align_distributions(
    words_a: &[String],
    probs_a: &[f64],
    words_b: &[String],
    probs_b: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    // Union vocabulary, smoothed
    let mut vocab: Vec<String> = words_a.iter().chain(words_b.iter()).cloned().collect();
    vocab.sort();
    vocab.dedup();

    let smoothing = 1e-10;
    let lookup_a: HashMap<&str, f64> = words_a.iter().zip(probs_a).map(|(w, &p)| (w.as_str(), p)).collect();
    let lookup_b: HashMap<&str, f64> = words_b.iter().zip(probs_b).map(|(w, &p)| (w.as_str(), p)).collect();

    let mut p = Vec::with_capacity(vocab.len());
    let mut q = Vec::with_capacity(vocab.len());
    for w in &vocab {
        p.push(*lookup_a.get(w.as_str()).unwrap_or(&0.0) + smoothing);
        q.push(*lookup_b.get(w.as_str()).unwrap_or(&0.0) + smoothing);
    }
    // Renormalize
    let sp: f64 = p.iter().sum();
    let sq: f64 = q.iter().sum();
    for x in &mut p { *x /= sp; }
    for x in &mut q { *x /= sq; }
    (p, q)
}

fn main() {
    let texts = [
        ("doc_a", "the cat sat on the mat"),
        ("doc_b", "the dog sat on the rug"),
        ("doc_c", "quantum computing uses qubits for parallel computation"),
    ];

    let tol = 1e-9;

    // Build distributions
    let dists: Vec<_> = texts.iter().map(|(_, t)| word_distribution(t)).collect();

    println!("Pairwise divergences between documents:\n");
    println!("{:<12} {:>8} {:>8} {:>8}", "Pair", "JS", "KL(a|b)", "KL(b|a)");
    println!("{}", "-".repeat(40));

    for i in 0..texts.len() {
        for j in (i + 1)..texts.len() {
            let (p, q) = align_distributions(&dists[i].0, &dists[i].1, &dists[j].0, &dists[j].1);
            let js = jensen_shannon_divergence(&p, &q, tol).unwrap();
            let kl_ab = kl_divergence(&p, &q, tol).unwrap();
            let kl_ba = kl_divergence(&q, &p, tol).unwrap();
            println!(
                "{:<12} {:>8.4} {:>8.4} {:>8.4}",
                format!("{}-{}", texts[i].0, texts[j].0),
                js,
                kl_ab,
                kl_ba
            );
        }
    }

    println!();
    println!("Entropy of each document (nats):");
    for (i, (name, _)) in texts.iter().enumerate() {
        let h = entropy_nats(&dists[i].1, tol).unwrap();
        println!("  {}: {:.4} nats ({} unique words)", name, h, dists[i].0.len());
    }

    println!();
    println!("Note: JS is symmetric and bounded [0, ln2]; KL is asymmetric and unbounded.");
    println!("Similar docs (a,b) have low JS; unrelated doc (c) has high JS against both.");
}

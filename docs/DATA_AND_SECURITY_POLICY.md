# Data distribution and provenance

This snapshot distributes package identifiers, inherited labels, split/component assignments, aggregate size metadata, implementation code, frozen protocols, and saved aggregate results.

Raw benign/malicious package archives, extracted package source, third-party dataset mirrors, payloads, model weights, generated graphs, sample prediction vectors, and bootstrap draws remain outside the distribution. No sample URL needs to be visited for the integrity check. Source-dataset repository references in the README identify upstream collections and do not grant redistribution rights to their contents.

The retained analysis is static: installing/importing/executing package samples is not required by the integrity verifier. Code files under `scripts/` implement the research pipeline; they are not malicious package samples. They are provided for source inspection and separately configured reproduction.

The original anonymous review repository and the full research workspace are separate from this curated snapshot. File selection uses the old checksum allowlist plus explicitly identified aggregate/code additions, not a recursive workspace upload. No Git history, token, private configuration, or raw sample is intentionally included. See `metadata/SOURCE_COPY_MAP.csv` for distribution provenance.

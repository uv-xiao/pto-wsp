# Impl: Artifact Manifest Schema

## Goals

- stable integration contract
- reproducibility and auditability
- backend-specific extension fields without breaking core readers

## Core fields (recommended)

- `htp_version`, `git_hash`, `build_env`
- `inputs`:
  - entrypoints
  - enabled dialects
  - intrinsic sets
- `target`:
  - backend name
  - hardware profile
- `pipeline`:
  - pass list with versions and parameters
- `outputs`:
  - emitted files with semantic roles
  - entry symbols and callable signatures

## Extensibility

- `extensions` namespace for backend/dialect-specific structured fields.

---

## Minimal example (illustrative JSON)

```json
{
  "schema": "htp.manifest.v1",
  "htp_version": "0.1.0-dev",
  "git_hash": "<repo-hash>",
  "build_env": {
    "python": "3.11.7",
    "platform": "linux-x86_64"
  },
  "inputs": {
    "entrypoints": [
      {"kind": "workload", "name": "add"},
      {"kind": "kernel", "name": "add_tile"}
    ],
    "dialects": ["wsp"],
    "intrinsic_sets": ["portable", "pto"]
  },
  "target": {
    "backend": "pto",
    "variant": "a2a3sim",
    "hardware_profile": "ascend:<profile-id>"
  },
  "capabilities": [
    "Dialect.WSPEnabled",
    "Layout.FacetSupported(dist)",
    "Layout.FacetSupported(mem)",
    "Backend.PTO(variant=a2a3sim)"
  ],
  "pipeline": {
    "name": "pto_default",
    "passes": [
      {"name": "ast_canonicalize", "version": "1"},
      {"name": "typecheck_layout_effects", "version": "1"},
      {"name": "apply_schedule", "version": "1"},
      {"name": "lower_pto", "version": "1"},
      {"name": "emit_pto_package", "version": "1"}
    ]
  },
  "outputs": {
    "package_manifest": "codegen/pto/package_manifest.json",
    "kernel_registry": "codegen/pto/kernel_registry.json",
    "entrypoints": [
      {"name": "add", "kind": "workload", "signature": "(*inputs) -> None"}
    ]
  },
  "extensions": {
    "pto": {
      "toolchain": "cann:<ver>",
      "runtime_contract": "pto-runtime:<ver>"
    }
  }
}
```


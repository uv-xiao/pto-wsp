# Impl: Testing Strategy (for compiler-as-artifact)

## Goals

- ensure stable artifact contracts
- ensure diagnostics are actionable and stable

## Recommended test classes

- Golden artifacts:
  - compile known examples
  - assert manifest fields and key files exist
- Type-check errors:
  - known-bad programs emit expected error kinds
- Backend packaging validation:
  - package validates against backend contract schema


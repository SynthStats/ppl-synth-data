"""Observability: OTel projection, span tree construction, redaction.

The local event journal is the single source of truth. Everything in this
package is a projection: we emit spans via the OTel SDK to a collector, which
fans out to Langfuse, Phoenix, Opik, or any OTLP-compatible backend. No
vendor-specific code sits on the hot path.
"""

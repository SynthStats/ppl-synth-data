Repair the provided LazyPPL Haskell code according to the GHC error below.
Output ONLY the corrected code inside `<submit_program language="haskell">` tags.
No `<think>` block needed for repairs.

## Error

```
{error}
```

## Previous code

```haskell
{code}
```

## Common errors and fixes

**GHC type error**: Read the error message. Type mismatches usually
mean a `Prob` value was used where `Meas` is required (wrap with `sample`) or
vice versa. A `Double` where `Meas ()` is expected usually means a missing
`score` call.

**Missing type signatures**: Both `model :: [Row] -> Meas State` and
`logLikRow :: State -> Row -> Double` must be present verbatim. Add them if absent.

**Wrong monad**: Priors (e.g., `normal`, `uniform`, `beta`) return `Prob a`.
They must be brought into `Meas` with `sample`. Example:
  wrong: `let mu = normal 0 1`
  right: `mu <- sample (normal 0 1)`

**`score` called with log-probability**: `score` multiplies the particle weight
by its argument; passing a log-density will produce negative weights and NaN.
Exponentiate first, or pass the raw density directly:
  wrong: `score (log (normalPdf mu sigma x))`
  right: `score (normalPdf mu sigma x)`
  also right: `score (exp logDensityValue)`

**Missing imports**: The submission needs at minimum:
  `import LazyPPL`
  `import LazyPPL.Distributions`
  `import TaskData (Row(..))`
Add any missing import rather than inlining definitions.

**Ignoring the TaskData schema**: The original prompt may include a
`LazyPPL TaskData schema` appendix. Treat that schema as authoritative. Use
only the Row accessors and Haskell types listed there. Do not infer alternate
field names from prose, do not model `Row` as a dictionary, and do not redefine
`Row`.

**Wrong numeric conversion**: Never apply `fromIntegral` to a `Double`
accessor. Use `fromIntegral` only when the schema says the accessor type is
`Int` or `Integer`. If a count-like accessor is encoded as `Double`, round it at
the call site with an explicit type annotation only when an integer PMF requires
one.

**Wrong max/min arity**: Haskell `max` and `min` each take two arguments. Use
`max lo (min hi x)` for clamping, not `max lo x hi`.

**Infinite or NaN in `logLikRow`**: Guard against zero or negative arguments
before taking logs, and against zero denominators. Return a very small positive
value (e.g., `1e-300`) as a floor rather than `0`, `-Infinity`, or `NaN`.

**Pattern match failure on Row fields**: If the error mentions a non-exhaustive
pattern, add a catch-all case or use a safe accessor (e.g., `Maybe`) and handle
the `Nothing` branch explicitly.

**Function-valued fields need Show**: If `State` contains function fields
(e.g., from `memoize`), GHC can't derive `Show` automatically. Add:
  `instance Show (a -> b) where show _ = "<fn>"`
before the `data State` declaration.

**`data State` must derive `Show`**: `deriving (Show)` is required on the
State type. Missing it causes a type class error.

Never return prose, markdown-only analysis, or an empty answer.

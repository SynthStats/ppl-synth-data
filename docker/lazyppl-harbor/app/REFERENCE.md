# LazyPPL Reference for Code Generation

LazyPPL is a Haskell library for Bayesian probabilistic programming with
lazy evaluation. Lazy evaluation allows non-parametric models (infinite streams,
Gaussian processes, Dirichlet processes) without truncation.

## Two-monad architecture

| Monad | Meaning | Key operations |
|-------|---------|----------------|
| `Prob a` | Normalized probability distribution over `a` | `uniform`, distribution constructors |
| `Meas a` | Unnormalized measure over `a` | `sample`, `score` |

`Prob` is for priors and generative models. `Meas` is for conditioning
(Bayesian inference): you `sample` from a prior, then `score` to weight
by likelihood.

```haskell
newtype Prob a = Prob (Tree -> a)       -- deterministic given randomness source
newtype Meas a = Meas (WriterT (Product (Log Double)) Prob a)
```

Both are `Monad` instances, so you compose with `do`-notation.

## Core API (module `LazyPPL`)

```haskell
-- The single primitive distribution: Uniform(0,1)
uniform :: Prob Double

-- Lift a Prob into Meas (sample from a prior)
sample :: Prob a -> Meas a

-- Weight the current execution by a likelihood (positive Double)
-- A score of 0 encodes impossibility (internally exp(-300)).
score :: Double -> Meas ()

-- Metropolis-Hastings sampler
-- p = probability of mutating each site per step
mh :: Double -> Meas a -> IO [(a, Product (Log Double))]

-- Irreducible MH (randomly restarts with probability q)
-- p = per-site mutation probability, q = full-restart probability
mhirreducible :: Double -> Double -> Meas a -> IO [(a, Product (Log Double))]

-- Weighted importance sampling (n reference samples)
wis :: Int -> Meas a -> IO [a]

-- Thin a stream: keep every n-th element
every :: Int -> [a] -> [a]

-- Run a Prob deterministically given a Tree
runProb :: Prob a -> Tree -> a

-- Generate a random Tree from a RandomGen seed
randomTree :: RandomGen g => g -> Tree
```

Return type of `mh` and `mhirreducible`: a lazy `[(a, Product (Log Double))]`
where `Product (Log Double)` is the unnormalized weight. Use `getProduct` to
extract the `Log Double`, and `ln` to get the raw log-value.

## Distributions (module `LazyPPL.Distributions`)

```haskell
-- Continuous
normal       :: Double -> Double -> Prob Double          -- (mu, sigma)
normalPdf    :: Double -> Double -> Double -> Double     -- (mu, sigma, x) -> density
exponential  :: Double -> Prob Double                    -- (rate)
expPdf       :: Double -> Double -> Double               -- (rate, x) -> density
gamma        :: Double -> Double -> Prob Double           -- (shape k, scale theta)
beta         :: Double -> Double -> Prob Double           -- (alpha, beta)
uniformbounded :: Double -> Double -> Prob Double         -- (lower, upper)
dirichlet    :: [Double] -> Prob [Double]                 -- (alphas) -> simplex vector

-- Discrete
bernoulli    :: Double -> Prob Bool                       -- (bias)
uniformdiscrete :: Int -> Prob Int                        -- (n) -> {0..n-1}
categorical  :: [Double] -> Prob Int                      -- (weights summing to 1) -> index
poisson      :: Double -> Prob Integer                    -- (rate lambda)
poissonPdf   :: Double -> Integer -> Double               -- (rate, k) -> pmf

-- Streams
iid          :: Prob a -> Prob [a]                        -- infinite i.i.d. stream
```

### Log-density functions (for `logLikRow`)

```haskell
normalLogPdf      :: Double -> Double -> Double -> Double   -- (mu, sigma, x) -> log-density
exponentialLogPdf  :: Double -> Double -> Double             -- (rate, x) -> log-density
gammaLogPdf        :: Double -> Double -> Double -> Double   -- (shape, scale, x) -> log-density
betaLogPdf         :: Double -> Double -> Double -> Double   -- (alpha, beta, x) -> log-density
bernoulliLogPmf    :: Double -> Bool -> Double               -- (p, x) -> log-pmf
categoricalLogPmf  :: [Double] -> Int -> Double              -- (probs, k) -> log-pmf
uniformBoundedLogPdf :: Double -> Double -> Double -> Double -- (lo, hi, x) -> log-density
poissonLogPmf      :: Double -> Integer -> Double            -- (rate, k) -> log-pmf
dirichletLogPdf    :: [Double] -> [Double] -> Double         -- (alphas, xs) -> log-density
logNormalLogPdf    :: Double -> Double -> Double -> Double   -- (mu, sigma, x) -> log-density
binomialLogPmf     :: Int -> Double -> Int -> Double         -- (n, p, k) -> log-pmf
```

Use log-density functions in `logLikRow` for numerical stability.
Use raw density functions (e.g. `normalPdf`) with `score` in `model`.

## Non-parametric constructs

### Stochastic memoization (`LazyPPL.Distributions.Memoization`)

```haskell
-- Memoize a random function: sample once per argument, lazily
memoize :: MonadMemo m a => (a -> m b) -> m (a -> b)

-- General memoization for Prob (works with any Ord argument)
generalmemoize :: Ord a => (a -> Prob b) -> Prob (a -> b)

-- Memoized fixed-point combinator for recursive random functions
memrec :: (Ord a, Show a) => ((a -> b) -> (a -> Prob b)) -> Prob (a -> b)
```

`memoize` converts `(a -> Prob b)` into `Prob (a -> b)`, sampling each
argument value at most once.

### Dirichlet process (`LazyPPL.Distributions.DirichletP`)

```haskell
-- Chinese Restaurant Process interface
newRestaurant :: Double -> Prob Restaurant       -- concentration alpha
newCustomer   :: Restaurant -> Prob Table         -- assign customer to table
-- Table supports Eq (same-cluster test) and MonadMemo Prob

-- Random distribution interface
dp :: Double -> Prob a -> Prob (Prob a)            -- (alpha, base) -> random distribution
```

### Gaussian processes (`LazyPPL.Distributions.GP`)

```haskell
wiener :: Prob (Double -> Double)                            -- Brownian motion
gp     :: (Double -> Double -> Double) -> Prob (Double -> Double)  -- GP with covariance kernel
```

Both return random functions. The GP uses a hidden memo table internally.

## Submission contract

Every submission must define exactly:

```haskell
module Submission where
import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

-- Parameters inferred by the model
data State = State { ... } deriving (Show)

-- Prior + scoring: define priors over State, score against observed rows
model :: [Row] -> Meas State
model rows = do
  -- sample priors
  -- score against each row
  ...

-- Pointwise log-likelihood for one observation given parameters
logLikRow :: State -> Row -> Double
logLikRow state row = ...
```

The harness (Main.hs) handles inference, seeding, burn-in, thinning, and
JSON output. Submissions must NOT call `mh`/`mhirreducible` or do any I/O.

## Worked examples

### 1. Coin bias estimation (Beta-Bernoulli)

```haskell
data State = State { theta :: Double } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  t <- sample $ beta 2 2
  mapM_ (\r -> score $ if heads r then t else (1 - t)) rows
  return $ State t

logLikRow :: State -> Row -> Double
logLikRow (State t) r =
  log (if heads r then t else (1 - t))
```

### 2. Normal with unknown mean and variance

```haskell
data State = State { mu :: Double, sigma :: Double } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  m  <- sample $ normal 0 10
  s  <- sample $ gamma 2 1
  mapM_ (\r -> score $ normalPdf m s (value r)) rows
  return $ State m s

logLikRow :: State -> Row -> Double
logLikRow (State m s) r =
  normalLogPdf m s (value r)
```

### 3. Linear regression

```haskell
data State = State
  { slopeA :: Double, interceptB :: Double, noiseS :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  a <- sample $ normal 0 3
  b <- sample $ normal 0 3
  s <- sample $ gamma 2 1
  mapM_ (\r -> score $ normalPdf (a * xVal r + b) s (yVal r)) rows
  return $ State a b s

logLikRow :: State -> Row -> Double
logLikRow (State a b s) r =
  normalLogPdf (a * xVal r + b) s (yVal r)
```

### 4. Poisson rate estimation

```haskell
data State = State { lambda :: Double } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  lam <- sample $ gamma 2 2
  mapM_ (\r -> score $ poissonPdf lam (count r)) rows
  return $ State lam

logLikRow :: State -> Row -> Double
logLikRow (State lam) r =
  poissonLogPmf lam (count r)
```

### 5. Mixture of two normals

```haskell
data State = State
  { mu1 :: Double, mu2 :: Double
  , sigma1 :: Double, sigma2 :: Double
  , mixWeight :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  m1 <- sample $ normal 0 10
  m2 <- sample $ normal 0 10
  s1 <- sample $ gamma 2 1
  s2 <- sample $ gamma 2 1
  w  <- sample $ beta 2 2
  mapM_ (\r ->
    let x = value r
        lik = w * normalPdf m1 s1 x + (1 - w) * normalPdf m2 s2 x
    in score lik) rows
  return $ State m1 m2 s1 s2 w

logLikRow :: State -> Row -> Double
logLikRow (State m1 m2 s1 s2 w) r =
  let x  = value r
      la = log w + normalLogPdf m1 s1 x
      lb = log (1 - w) + normalLogPdf m2 s2 x
      mx = max la lb
  in mx + log (exp (la - mx) + exp (lb - mx))
```

### 6. Exponential survival model

```haskell
data State = State { rate :: Double } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  r <- sample $ gamma 2 1
  mapM_ (\row -> score $ expPdf r (duration row)) rows
  return $ State r

logLikRow :: State -> Row -> Double
logLikRow (State r) row =
  exponentialLogPdf r (duration row)
```

### 7. Multivariate regression (two predictors)

```haskell
data State = State
  { b0 :: Double, b1 :: Double, b2 :: Double, noiseSigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  b0' <- sample $ normal 0 5
  b1' <- sample $ normal 0 5
  b2' <- sample $ normal 0 5
  s   <- sample $ gamma 2 1
  mapM_ (\r ->
    let predicted = b0' + b1' * x1Val r + b2' * x2Val r
    in score $ normalPdf predicted s (yVal r)) rows
  return $ State b0' b1' b2' s

logLikRow :: State -> Row -> Double
logLikRow (State b0' b1' b2' s) r =
  let predicted = b0' + b1' * x1Val r + b2' * x2Val r
  in normalLogPdf predicted s (yVal r)
```

### 8. Non-parametric clustering (Dirichlet process)

```haskell
import LazyPPL.Distributions.DirichletP (newRestaurant, newCustomer)
import LazyPPL.Distributions.Memoization (memoize)

data State = State { assignments :: [Int] } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  rest  <- sample $ newRestaurant 1.0
  muFn  <- sample $ memoize $ const $ normal 0 10
  sigFn <- sample $ memoize $ const $ gamma 2 1
  assigns <- mapM (\r -> do
    table <- sample $ newCustomer rest
    score $ normalPdf (muFn table) (sigFn table) (value r)
    return table) rows
  return $ State assigns

-- For non-parametric models, logLikRow must marginalize over
-- cluster assignments. A simple approximation: use the assigned
-- cluster's parameters directly.
logLikRow :: State -> Row -> Double
logLikRow _ _ = error "DP clustering: requires marginalizing over assignments"
```

### 9. Logistic regression (binary outcome)

```haskell
data State = State
  { beta0 :: Double, beta1 :: Double
  } deriving (Show)

sigmoid :: Double -> Double
sigmoid z = 1.0 / (1.0 + exp (-z))

model :: [Row] -> Meas State
model rows = do
  b0 <- sample $ normal 0 5
  b1 <- sample $ normal 0 5
  mapM_ (\r ->
    let p = sigmoid (b0 + b1 * xVal r)
        lik = if label r then p else (1 - p)
    in score lik) rows
  return $ State b0 b1

logLikRow :: State -> Row -> Double
logLikRow (State b0 b1) r =
  let p = sigmoid (b0 + b1 * xVal r)
  in log (if label r then p else (1 - p))
```

### 10. Hierarchical model (group-level effects)

```haskell
import LazyPPL.Distributions.Memoization (memoize)

data State = State
  { globalMu :: Double, globalSigma :: Double, groupSigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  gMu    <- sample $ normal 0 10
  gSigma <- sample $ gamma 2 1
  sSigma <- sample $ gamma 2 1
  groupMu <- sample $ memoize $ const $ normal gMu gSigma
  mapM_ (\r ->
    score $ normalPdf (groupMu (groupId r)) sSigma (value r)) rows
  return $ State gMu gSigma sSigma

logLikRow :: State -> Row -> Double
logLikRow _ _ = error "requires group-level parameters; use full model weight"
```

## Common patterns

**Scoring against data**: loop over rows with `mapM_` and `score` by the
likelihood (a positive `Double`, not a log-likelihood).

**Priors**: use `sample $ <distribution>` in `Meas`. Use `<-` directly
in `Prob`.

**Infinite structures**: use `iid`, `memoize`, or direct recursion.
Laziness handles truncation.

**Typical MH call** (in the harness, not submissions):
```haskell
samples <- mhirreducible 0.3 0.05 (model rows)
let results = map fst $ take 2000 $ every 100 $ drop 5000 samples
```

## What NOT to do in submissions

- Don't call `mh`, `mhirreducible`, `wis`, or any IO actions
- No `unsafePerformIO`
- No graphics library imports
- No file reads or network I/O
- Don't define `main`
- `score` takes a positive `Double` (the likelihood), not a log-likelihood
- Priors should be weakly informative, not hardcoded to data values

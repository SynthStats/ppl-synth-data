module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import LazyPPL.Distributions.Memoization (memoize)
import TaskData (Row(..))

-- Per-type PK parameters and per-subject type assignment
data State = State
  { ke1 :: Double, ke2 :: Double, ke3 :: Double
  , ka1 :: Double, ka2 :: Double, ka3 :: Double
  , amp1 :: Double, amp2 :: Double, amp3 :: Double
  , sigmaProp :: Double
  , subjectType :: Int -> Int  -- subject index -> metabolizer type (0,1,2)
  } deriving (Show)

instance Show (Int -> Int) where show _ = "<fn>"

-- One-compartment concentration at time t
conc :: Double -> Double -> Double -> Double -> Double
conc ka' ke' amp' t =
  let ratio = ka' / (ka' - ke')
  in amp' * ratio * (exp (-ke' * t) - exp (-ka' * t))

model :: [Row] -> Meas State
model rows = do
  probs <- sample $ dirichlet [30, 50, 20]
  let [p0, p1, p2] = probs

  -- ke = log(2)/half_life; half-lives ~2, 4, 8 hrs
  logKe1 <- sample $ normal (log 0.35) 0.3
  logKe2 <- sample $ normal (log 0.17) 0.3
  logKe3 <- sample $ normal (log 0.087) 0.3
  let ke1' = exp logKe1
      ke2' = exp logKe2
      ke3' = exp logKe3

  -- ka > ke
  dka1 <- sample $ gamma 2 0.5
  dka2 <- sample $ gamma 2 0.25
  dka3 <- sample $ gamma 2 0.15
  let ka1' = ke1' + dka1
      ka2' = ke2' + dka2
      ka3' = ke3' + dka3

  logAmp1 <- sample $ normal (log 60) 0.4
  logAmp2 <- sample $ normal (log 50) 0.4
  logAmp3 <- sample $ normal (log 40) 0.4
  let amp1' = exp logAmp1
      amp2' = exp logAmp2
      amp3' = exp logAmp3

  sProp <- sample $ gamma 2 0.075

  -- discrete latent: fails in PyMC NUTS, works in LazyPPL MH
  typeFn <- sample $ memoize $ const $ categorical probs

  mapM_ (\r -> do
    let sid = subjectId r
        t   = time r
        typ = typeFn sid
        (ka', ke', amp') = case typ of
          0 -> (ka1', ke1', amp1')
          1 -> (ka2', ke2', amp2')
          _ -> (ka3', ke3', amp3')
        pred = max 1e-6 (conc ka' ke' amp' t)
        sig  = sProp * pred + 0.5
    score $ normalPdf pred sig (concentration r)
    ) rows

  return $ State ke1' ke2' ke3' ka1' ka2' ka3'
                 amp1' amp2' amp3' sProp typeFn

logLikRow :: State -> Row -> Double
logLikRow st r =
  let sid = subjectId r
      t   = time r
      typ = subjectType st sid
      (ka', ke', amp') = case typ of
        0 -> (ka1 st, ke1 st, amp1 st)
        1 -> (ka2 st, ke2 st, amp2 st)
        _ -> (ka3 st, ke3 st, amp3 st)
      pred = max 1e-6 (conc ka' ke' amp' t)
      sig  = sigmaProp st * pred + 0.5
  in normalLogPdf pred sig (concentration r)

module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { piDegraded :: Double
  , pStayNormal :: Double
  , pStayDegraded :: Double
  , meanNormal :: Double
  , meanDegraded :: Double
  , concNormal :: Double
  , concDegraded :: Double
  , latentPath :: [Bool]
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  piD <- sample (beta 1 1)
  pSN <- sample (beta 12 1.5)
  pSD <- sample (beta 12 1.5)

  mN <- sample (beta 1.5 40)
  meanGap <- sample (beta 2 6)
  let mD = clamp01 (mN + 0.05 + 0.45 * meanGap)

  cN <- sample (gamma 2 40)
  cD <- sample (gamma 2 20)

  let aNormal = safeShape (mN * cN)
      bNormal = safeShape ((1 - mN) * cN)
      aDegraded = safeShape (mD * cD)
      bDegraded = safeShape ((1 - mD) * cD)

  states <- go piD pSN pSD aNormal bNormal aDegraded bDegraded rows Nothing []
  let path = reverse states

  return
    State
      { piDegraded = piD
      , pStayNormal = pSN
      , pStayDegraded = pSD
      , meanNormal = mN
      , meanDegraded = mD
      , concNormal = cN
      , concDegraded = cD
      , latentPath = path
      }
  where
    go :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> [Row] -> Maybe Bool -> [Bool] -> Meas [Bool]
    go _ _ _ _ _ _ _ [] _ acc = return acc
    go piD pSN pSD aNormal bNormal aDegraded bDegraded (r:rs) mPrev acc = do
      z <-
        case mPrev of
          Nothing -> sample (bernoulli piD)
          Just prev ->
            if prev
              then sample (bernoulli pSD)
              else sample (bernoulli (1 - pSN))

      let x = clamp01 (errorRate r)
          ll =
            if z
              then betaLogPdf aDegraded bDegraded x
              else betaLogPdf aNormal bNormal x

      score (exp ll)
      go piD pSN pSD aNormal bNormal aDegraded bDegraded rs (Just z) (z : acc)

    clamp01 :: Double -> Double
    clamp01 x =
      let eps = 1e-9
       in max eps (min (1 - eps) x)

    safeShape :: Double -> Double
    safeShape x = max 1e-6 x

logLikRow :: State -> Row -> Double
logLikRow st row =
  let x = clamp01 (errorRate row)

      aN = safeShape (meanNormal st * concNormal st)
      bN = safeShape ((1 - meanNormal st) * concNormal st)
      aD = safeShape (meanDegraded st * concDegraded st)
      bD = safeShape ((1 - meanDegraded st) * concDegraded st)

      wD = clampProb (piDegraded st)
      wN = clampProb (1 - wD)

      llN = betaLogPdf aN bN x
      llD = betaLogPdf aD bD x

      m = max llN llD
      mix = wN * exp (llN - m) + wD * exp (llD - m)
   in if mix <= 0 || isNaN mix || isInfinite mix
        then -1e300
        else m + log mix
  where
    clamp01 :: Double -> Double
    clamp01 y =
      let eps = 1e-9
       in max eps (min (1 - eps) y)

    clampProb :: Double -> Double
    clampProb p =
      let eps = 1e-12
       in max eps (min (1 - eps) p)

    safeShape :: Double -> Double
    safeShape y = max 1e-6 y
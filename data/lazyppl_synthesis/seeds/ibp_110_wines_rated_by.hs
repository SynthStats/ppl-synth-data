module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

instance Show (a -> b) where
  show _ = "<fn>"

data State = State
  { stateMu :: Double
  , stateSigma :: Double
  , stateWineBias :: Double -> Double
  , stateCriticBias :: Double -> Double
  , stateWineProfile :: Double -> [Double]
  , stateCriticWeight :: Double -> [Double]
  , stateNumProfiles :: Int
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  let k = 4
      eps = 1e-6
      positive x = if x > eps then x else eps
      dot xs ys = sum (zipWith (*) xs ys)

      wineIds = unique (map rowWine rows)
      criticIds = unique (map rowCritic rows)

  mu <- sample (normal 75.0 15.0)
  sigma <- sample (gamma 2.0 3.0)
  sigmaWine <- sample (gamma 2.0 1.0)
  sigmaCritic <- sample (gamma 2.0 1.0)
  sigmaWineBias <- sample (gamma 2.0 2.0)
  sigmaCriticBias <- sample (gamma 2.0 2.0)

  wineBiasParams <- mapM (\_ -> sample (normal 0.0 (positive sigmaWineBias))) wineIds
  criticBiasParams <- mapM (\_ -> sample (normal 0.0 (positive sigmaCriticBias))) criticIds
  wineProfileParams <- mapM (\_ -> mapM (\_ -> sample (normal 0.0 (positive sigmaWine))) [1 .. k]) wineIds
  criticWeightParams <- mapM (\_ -> mapM (\_ -> sample (normal 0.0 (positive sigmaCritic))) [1 .. k]) criticIds

  let lookupWithDefault key pairs def =
        case lookup key pairs of
          Just v -> v
          Nothing -> def

      zeroVec = replicate k 0.0

      wineBiasF w = lookupWithDefault w (zip wineIds wineBiasParams) 0.0
      criticBiasF c = lookupWithDefault c (zip criticIds criticBiasParams) 0.0
      wineProfileF w = lookupWithDefault w (zip wineIds wineProfileParams) zeroVec
      criticWeightF c = lookupWithDefault c (zip criticIds criticWeightParams) zeroVec

      st =
        State
          { stateMu = mu
          , stateSigma = positive sigma
          , stateWineBias = wineBiasF
          , stateCriticBias = criticBiasF
          , stateWineProfile = wineProfileF
          , stateCriticWeight = criticWeightF
          , stateNumProfiles = k
          }

      boundedMean x
        | x < 50.0 = 50.0
        | x > 100.0 = 100.0
        | otherwise = x

      rowWeight row =
        let w = rowWine row
            c = rowCritic row
            y = rowRating row
            wb = stateWineBias st w
            cb = stateCriticBias st c
            wp = stateWineProfile st w
            cw = stateCriticWeight st c
            meanRating = boundedMean (mu + wb + cb + dot wp cw)
            dens = normalPdf meanRating (positive sigma) y
         in if dens > eps then dens else eps

  mapM_ (\row -> score (rowWeight row)) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let eps = 1e-300
      positive x = if x > eps then x else eps
      dot xs ys = sum (zipWith (*) xs ys)
      w = rowWine row
      c = rowCritic row
      y = rowRating row
      mu = stateMu st
      sigma = positive (stateSigma st)
      wb = stateWineBias st w
      cb = stateCriticBias st c
      wp = stateWineProfile st w
      cw = stateCriticWeight st c
      meanRating0 = mu + wb + cb + dot wp cw
      meanRating
        | meanRating0 < 50.0 = 50.0
        | meanRating0 > 100.0 = 100.0
        | otherwise = meanRating0
      dens = normalPdf meanRating sigma y
   in log (positive dens)

unique :: Eq a => [a] -> [a]
unique = foldr (\x acc -> if x `elem` acc then acc else x : acc) []
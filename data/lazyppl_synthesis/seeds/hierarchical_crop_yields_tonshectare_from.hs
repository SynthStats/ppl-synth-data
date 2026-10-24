module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { stateGlobalMean :: Double
  , stateSigmaRegion :: Double
  , stateSigmaFarm :: Double
  , stateSigmaYear :: Double
  , stateSigmaObs :: Double
  , stateRegionMeans :: [Double]
  , stateFarmOffsets :: [Double]
  , stateYearEffects :: [Double]
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  globalMean <- sample (normal 5.0 3.0)
  sigmaRegion0 <- sample (gamma 2.0 1.0)
  sigmaFarm0 <- sample (gamma 2.0 1.0)
  sigmaYear0 <- sample (gamma 2.0 1.0)
  sigmaObs0 <- sample (gamma 2.0 1.0)

  let sigmaRegion = sigmaRegion0 + 1e-3
      sigmaFarm = sigmaFarm0 + 1e-3
      sigmaYear = sigmaYear0 + 1e-3
      sigmaObs = sigmaObs0 + 1e-3

      regionIds = map (round . rowRegionIndex) rows
      farmIds = map (round . rowFarmIndex) rows
      yearIds = map (round . rowYearIndex) rows

      nRegions =
        if null regionIds then 0 else maximum regionIds + 1
      nFarms =
        if null farmIds then 0 else maximum farmIds + 1
      nYears =
        if null yearIds then 0 else maximum yearIds + 1

  regionMeans <- sample $
    sequence
      [ normal globalMean sigmaRegion
      | _ <- [0 .. nRegions - 1]
      ]

  farmOffsets <- sample $
    sequence
      [ normal 0.0 sigmaFarm
      | _ <- [0 .. nFarms - 1]
      ]

  yearEffects <- sample $
    sequence
      [ normal 0.0 sigmaYear
      | _ <- [0 .. nYears - 1]
      ]

  let st =
        State
          { stateGlobalMean = globalMean
          , stateSigmaRegion = sigmaRegion
          , stateSigmaFarm = sigmaFarm
          , stateSigmaYear = sigmaYear
          , stateSigmaObs = sigmaObs
          , stateRegionMeans = regionMeans
          , stateFarmOffsets = farmOffsets
          , stateYearEffects = yearEffects
          }

  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let y = rowYield row
      ri = round (rowRegionIndex row)
      fi = round (rowFarmIndex row)
      yi = round (rowYearIndex row)

      regionMeans = stateRegionMeans st
      farmOffsets = stateFarmOffsets st
      yearEffects = stateYearEffects st

      safeIndex xs k def =
        if k < 0 || k >= length xs then def else xs !! k

      mu =
        safeIndex regionMeans ri (stateGlobalMean st)
          + safeIndex farmOffsets fi 0.0
          + safeIndex yearEffects yi 0.0

      sigma = max 1e-6 (stateSigmaObs st)
   in normalLogPdf mu sigma y
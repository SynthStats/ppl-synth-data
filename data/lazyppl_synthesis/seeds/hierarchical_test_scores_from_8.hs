module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { globalMean :: Double
  , districtSd :: Double
  , schoolSd :: Double
  , withinSd :: Double
  , districtMeansState :: [(Double, Double)]
  , schoolMeansState :: [(Double, Double)]
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  let eps = 1e-6
      uniq :: Eq a => [a] -> [a]
      uniq [] = []
      uniq (x:xs) = x : uniq [y | y <- xs, y /= x]

      districtIds = uniq [districtId r | r <- rows]
      schoolPairs = uniq [(schoolId r, districtId r) | r <- rows]

  muGlobal <- sample (normal 500 150)
  sigmaDistrictRaw <- sample (gamma 2 50)
  sigmaSchoolRaw <- sample (gamma 2 30)
  sigmaWithinRaw <- sample (gamma 2 30)

  let sigmaDistrict = max eps sigmaDistrictRaw
      sigmaSchool = max eps sigmaSchoolRaw
      sigmaWithin = max eps sigmaWithinRaw

  districtMeans <-
    sequence
      [ do
          m <- sample (normal muGlobal sigmaDistrict)
          return (d, m)
      | d <- districtIds
      ]

  let lookupDistrictMean d =
        case [m | (d', m) <- districtMeans, d' == d] of
          (m:_) -> m
          [] -> muGlobal

  schoolMeans <-
    sequence
      [ do
          let md = lookupDistrictMean d
          m <- sample (normal md sigmaSchool)
          return (s, m)
      | (s, d) <- schoolPairs
      ]

  let lookupSchoolMean s =
        case [m | (s', m) <- schoolMeans, s' == s] of
          (m:_) -> m
          [] -> muGlobal

      st =
        State
          { globalMean = muGlobal
          , districtSd = sigmaDistrict
          , schoolSd = sigmaSchool
          , withinSd = sigmaWithin
          , districtMeansState = districtMeans
          , schoolMeansState = schoolMeans
          }

      rowLik r =
        let mu = lookupSchoolMean (schoolId r)
            y = scoreValue r
        in max eps (normalPdf mu sigmaWithin y)

  mapM_ (\r -> score (rowLik r)) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let eps = 1e-6
      muGlobal0 = globalMean st
      sigmaWithin0 = max eps (withinSd st)
      schoolMeans0 = schoolMeansState st
      muSchool =
        case [m | (s, m) <- schoolMeans0, s == schoolId row] of
          (m:_) -> m
          [] -> muGlobal0
      y = scoreValue row
      dens = max 1e-300 (normalPdf muSchool sigmaWithin0 y)
  in log dens
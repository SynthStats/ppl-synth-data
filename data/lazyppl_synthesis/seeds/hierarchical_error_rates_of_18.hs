module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { globalIntercept :: Double
  , modelEffects :: [Double]
  , taskEffects :: [Double]
  , modelSd :: Double
  , taskSd :: Double
  , precision :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  alpha <- sample (normal 0.0 1.5)
  sigmaModel0 <- sample (gamma 2.0 1.0)
  sigmaTask0  <- sample (gamma 2.0 1.0)
  phi0 <- sample (gamma 2.0 20.0)

  let sigmaModel = max 1e-6 sigmaModel0
      sigmaTask  = max 1e-6 sigmaTask0
      phi        = max 1e-6 phi0

      maxModel = maximum (0 : map modelId rows)
      maxTask  = maximum (0 : map taskId rows)

  modelEffs <- mapM (\_ -> sample (normal 0.0 sigmaModel)) [0 .. maxModel]
  taskEffs  <- mapM (\_ -> sample (normal 0.0 sigmaTask))  [0 .. maxTask]

  let state = State
        { globalIntercept = alpha
        , modelEffects = modelEffs
        , taskEffects = taskEffs
        , modelSd = sigmaModel
        , taskSd = sigmaTask
        , precision = phi
        }

  mapM_ (\r -> score (exp (logLikRow state r))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let mid = round (modelId row) :: Int
      tid = round (taskId row) :: Int
      y0  = errorRate row

      clip lo hi x = max lo (min hi x)
      y = clip 1e-6 (1.0 - 1e-6) y0

      alpha = globalIntercept state
      ms = modelEffects state
      ts = taskEffects state
      phi = max 1e-6 (precision state)

      me = if mid >= 0 && mid < length ms then ms !! mid else 0.0
      te = if tid >= 0 && tid < length ts then ts !! tid else 0.0

      eta = alpha + me + te
      mu0 = 1.0 / (1.0 + exp (-eta))
      mu = clip 1e-6 (1.0 - 1e-6) mu0

      a = max 1e-6 (mu * phi)
      b = max 1e-6 ((1.0 - mu) * phi)
  in betaLogPdf a b y
module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

instance Show (a -> b) where
  show _ = "<fn>"

data State = State
  { skillMasteryProb :: [Double]
  , skillRequireProb :: [Double]
  , masteryFn :: Int -> Int -> Bool
  , requireFn :: Int -> Int -> Bool
  , scoreBase :: Double
  , scoreGain :: Double
  , scoreSigma :: Double
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  let kSkills = 6
      skillIx = [0 .. kSkills - 1]
      nStudents = 90
      nQuizzes = 10

  masteryTables <-
    mapM
      (\_ -> mapM (\_ -> sample (bernoulli 0.5)) [0 .. nStudents - 1])
      skillIx

  requireTables <-
    mapM
      (\_ -> mapM (\_ -> sample (bernoulli 0.5)) [0 .. nQuizzes - 1])
      skillIx

  scoreBase <- sample (normal 50 20)
  rawGain <- sample (normal 30 20)
  scoreSigma <- sample (gamma 2 10)

  let masteryFn' sk sid =
        if sk >= 0 && sk < length masteryTables && sid >= 0 && sid < nStudents
          then (masteryTables !! sk) !! sid
          else False
      requireFn' sk qid =
        if sk >= 0 && sk < length requireTables && qid >= 0 && qid < nQuizzes
          then (requireTables !! sk) !! qid
          else False
      masteryProbs = [0.5 | _ <- skillIx]
      requireProbs = [0.5 | _ <- skillIx]
      scoreGain' = abs rawGain
      st =
        State
          { skillMasteryProb = masteryProbs
          , skillRequireProb = requireProbs
          , masteryFn = masteryFn'
          , requireFn = requireFn'
          , scoreBase = scoreBase
          , scoreGain = scoreGain'
          , scoreSigma = scoreSigma
          }

  mapM_ (\r -> score (max 1e-300 (exp (logLikRow st r)))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let sid = floor (rowStudent row)
      qid = floor (rowQuiz row)
      yRaw = rowScore row
      y = max 0 (min 100 yRaw)
      ks = [0 .. length (skillMasteryProb st) - 1]
      reqs = [if requireFn st k qid then 1.0 else 0.0 | k <- ks]
      hits =
        [ if requireFn st k qid && masteryFn st k sid then 1.0 else 0.0
        | k <- ks
        ]
      reqCount = sum reqs
      hitCount = sum hits
      frac =
        if reqCount <= 0
          then 0.5
          else hitCount / reqCount
      mu = scoreBase st + scoreGain st * frac
      sigma = max 1e-6 (scoreSigma st)
   in normalLogPdf mu sigma y
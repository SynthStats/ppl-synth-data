module TaskData
  ( Row(..)
  , loadData
  ) where

import System.IO.Error (ioError, userError)

data Row = Row
  deriving (Show)

loadData :: IO [Row]
loadData =
  ioError
    ( userError
        "TaskData.hs was not generated; export a Harbor task with visible rows before running the LazyPPL verifier."
    )

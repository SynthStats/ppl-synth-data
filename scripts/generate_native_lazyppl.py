#!/usr/bin/env python3
"""Generate native LazyPPL records for families that only exist in LazyPPL.

Eight native families:
- Stochastic memoization (12 records): entity-level effects, random graph coloring
- Gaussian Process (12 records): smooth function regression, spatial interpolation
- Changepoint / HMM (10 records): regime detection, structural breaks
- Dirichlet Process / CRP (10 records): non-parametric clustering
- Indian Buffet Process (12 records): sparse latent feature discovery
- Log-Gaussian Cox Process (8 records): spatial event intensity modeling
- Hierarchical models (12 records): multi-level pooling, partial pooling
- Point processes (10 records): event timing, self-exciting, thinning

Usage:
    uv run python scripts/generate_native_lazyppl.py \
        --output generated/lazyppl_synthesis/inputs/native_train.jsonl \
        --model gpt-5.5 \
        --max-repairs 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from litellm import completion as litellm_completion

from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.runtimes import get_runtime

LAZYPPL = get_runtime("lazyppl")
SYSTEM_PROMPT = load_prompt_text("completion_gen_lazyppl.md")

# prompts use natural language: no distribution-family names
NATIVE_PROMPTS: list[dict[str, str]] = [
    # --- STOCHASTIC MEMOIZATION (12) ---
    {
        "prompt": "A school district tracks student performance across 15 schools. Each school has "
        "a characteristic baseline achievement level. 200 students are tested, with each "
        "student belonging to one school. Test scores range 40-100. Schools with more "
        "resources tend to have higher baselines, but there is variation within each "
        "school.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "An art gallery has 30 painters, each with a distinctive price range for their "
        "works. We observe 150 painting sales. Each painting's price depends on the "
        "artist's typical price level plus random variation. Model the per-artist "
        "pricing from the sales data.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "A map coloring problem: 8 regions share borders (regions 0-1, 0-2, 1-3, 2-3, "
        "3-4, 4-5, 5-6, 6-7, 4-7 share borders). Each region is assigned one of 4 "
        "colors at random, but adjacent regions should avoid the same color. We "
        "observe partial assignments for regions 0, 3, and 7. Infer the full coloring.",
        "family": "memoization",
        "complexity": "complex",
    },
    {
        "prompt": "In an online game, 50 players each have a hidden skill level. When two players "
        "compete, the one with higher skill wins more often. We observe 200 match "
        "outcomes (winner and loser IDs). Skill levels are drawn independently for "
        "each player from a shared population distribution.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "A medical practice assigns patients to one of 20 doctors. Each doctor has their "
        "own treatment success rate. We observe 500 treatment outcomes (doctor ID and "
        "whether the treatment succeeded). Some doctors see more patients than others. "
        "Infer each doctor's effectiveness.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "A factory has 12 machines, each producing widgets. Each machine has a "
        "characteristic defect rate. We inspect 600 widgets and record which machine "
        "produced each and whether it was defective. Machine defect rates are drawn "
        "from a common prior. Infer per-machine defect rates.",
        "family": "memoization",
        "complexity": "simple",
    },
    {
        "prompt": "A language model assigns each word a hidden embedding dimension. When the same "
        "word appears multiple times, it gets the same embedding. We observe 100 word "
        "tokens from a vocabulary of 25 unique words, each with a noisy measurement "
        "of one embedding coordinate. Infer the per-word embedding value.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "A social network has 40 users. Each user has a hidden influence score. When "
        "user A mentions user B, user B is more likely to repost if A has high "
        "influence. We observe 300 mention-repost pairs. Model the per-user "
        "influence from repost data.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "A restaurant chain has 25 locations, each with a different average wait time. "
        "We observe 400 customer wait times with location IDs. Wait times within each "
        "location are noisy around the location's average, which is drawn from a "
        "shared prior. Infer location-level wait time distributions.",
        "family": "memoization",
        "complexity": "simple",
    },
    {
        "prompt": "A taxi fleet has 30 drivers. Each driver has a characteristic tip percentage "
        "they receive. We observe 500 rides with driver ID and tip percentage. Tips "
        "are noisy around each driver's baseline. Some drivers are more charming than "
        "others. Model the per-driver tipping behavior.",
        "family": "memoization",
        "complexity": "simple",
    },
    {
        "prompt": "A university has 10 departments. Each department has a hidden publication rate. "
        "We observe publication counts for 100 faculty members, each belonging to one "
        "department. Counts are noisy but faculty in the same department share similar "
        "rates. Model the department-level publication intensity.",
        "family": "memoization",
        "complexity": "medium",
    },
    {
        "prompt": "A random function maps city IDs (1-20) to hidden economic indicators. Each "
        "city's indicator is drawn independently but from a shared prior. We observe "
        "the indicator values for 8 cities with noise, and need to predict the "
        "remaining 12. The function value for each city is sampled once and reused.",
        "family": "memoization",
        "complexity": "complex",
    },
    # --- GAUSSIAN PROCESS (12) ---
    {
        "prompt": "Temperature readings at 30 weather stations along a 200km transect. Stations "
        "are unevenly spaced. Nearby stations show similar temperatures. Observations "
        "have measurement noise of about 0.5 degrees. Predict temperature at "
        "unobserved locations.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "A smooth function maps time (0-24 hours) to website traffic (page views per "
        "minute). We observe noisy traffic measurements at 40 time points. The "
        "underlying traffic pattern varies smoothly throughout the day with peaks "
        "and troughs.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "Ocean depth measurements at 25 locations in a bay. Depth varies smoothly with "
        "position (measured as distance from shore, 0-5km). Observations have sonar "
        "noise. Model the depth profile as a smooth function of distance.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "Air quality index readings at 35 monitoring points across a city, measured by "
        "distance from a factory (0-15km). AQI decreases smoothly with distance but "
        "measurements are noisy. Predict AQI at arbitrary distances.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "A materials science experiment: applied voltage (0-10V) produces a current "
        "response. 20 measurements taken at different voltages. The response is a "
        "smooth nonlinear function. Measurement noise is about 2% of the reading.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "Soil moisture content measured at 30 locations along a 500m hillside. Moisture "
        "varies smoothly with position but depends on local terrain. Readings have "
        "sensor noise. The underlying moisture function is continuous.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "Growth rate of a bacterial colony measured at 25 time points over 48 hours. "
        "The growth rate changes smoothly: starts low, peaks, then declines as resources "
        "deplete. Observations are noisy OD600 readings.",
        "family": "gp",
        "complexity": "medium",
    },
    {
        "prompt": "A cyclist's heart rate recorded at 20 points during a 2-hour ride over hilly "
        "terrain. Heart rate varies smoothly and correlates with the terrain's gradient. "
        "Model heart rate as a smooth function of elapsed time.",
        "family": "gp",
        "complexity": "simple",
    },
    {
        "prompt": "Annual rainfall measurements at 15 locations across a valley, measured by "
        "altitude (200-2000m). Rainfall increases smoothly with altitude but with "
        "significant noise from year-to-year variation. Model the altitude-rainfall "
        "relationship.",
        "family": "gp",
        "complexity": "simple",
    },
    {
        "prompt": "Concentration of a pollutant measured at 20 points downstream from a discharge "
        "point. Distance ranges 0-10km. Concentration decays smoothly but with "
        "measurement noise from sampling variability.",
        "family": "gp",
        "complexity": "simple",
    },
    {
        "prompt": "Light intensity measured at 30 positions across a greenhouse (0-50m). Some "
        "areas have supplemental lighting. Intensity varies smoothly with position. "
        "Sensor readings are noisy. Predict intensity at any position.",
        "family": "gp",
        "complexity": "simple",
    },
    {
        "prompt": "A robot arm's motor torque as a function of joint angle (0-180 degrees). "
        "35 measurements taken. Torque varies smoothly with angle in a nonlinear "
        "pattern. Friction causes measurement noise of about 5% of the true torque.",
        "family": "gp",
        "complexity": "medium",
    },
    # --- CHANGEPOINT / HMM (10) ---
    {
        "prompt": "Monthly sales of ice cream over 5 years (60 observations). Sales were steady "
        "around 500 units until a competitor entered the market, after which sales "
        "dropped to around 350 units. The exact month of the change is unknown. "
        "Sales have noise of about 50 units.",
        "family": "changepoint",
        "complexity": "medium",
    },
    {
        "prompt": "A sensor monitors vibration levels in a bridge, recorded hourly for 200 hours. "
        "Initially, vibrations are low (baseline ~2.0 mm/s). At some point, structural "
        "damage causes vibrations to shift to a higher level (~5.5 mm/s). Find when "
        "the damage occurred. Measurements have noise.",
        "family": "changepoint",
        "complexity": "medium",
    },
    {
        "prompt": "Heart rate data from a patient over 100 minutes at 1-minute intervals. The "
        "patient alternates between rest (~70 bpm) and stress (~100 bpm) states. State "
        "transitions happen at unknown times. Observations are noisy (SD ~5 bpm). "
        "Model the state-switching pattern.",
        "family": "changepoint",
        "complexity": "complex",
    },
    {
        "prompt": "Daily website error rates over 90 days. The system runs normally (~0.1% errors) "
        "but occasionally enters a degraded state (~2% errors) due to infrastructure "
        "issues. State transitions are rare but unpredictable. Model the error regime.",
        "family": "changepoint",
        "complexity": "complex",
    },
    {
        "prompt": "Stock price returns over 250 trading days. The market alternates between calm "
        "periods (low volatility, SD ~0.5%) and turbulent periods (high volatility, "
        "SD ~2%). Regime changes are abrupt. Model the volatility regime from returns.",
        "family": "changepoint",
        "complexity": "complex",
    },
    {
        "prompt": "Temperature readings from a chemical reactor over 150 time steps. The reactor "
        "operates in two modes: baseline (setpoint 180C, noise 3C) and overheating "
        "(setpoint 220C, noise 5C). A controller fault causes mode switches. Detect "
        "the switching points.",
        "family": "changepoint",
        "complexity": "medium",
    },
    {
        "prompt": "A call center receives calls over 120 hours. Call rate was ~20/hour until "
        "a marketing campaign began, increasing it to ~40/hour. The campaign start "
        "time is uncertain (somewhere in hours 30-60). Model the rate change from "
        "hourly counts.",
        "family": "changepoint",
        "complexity": "medium",
    },
    {
        "prompt": "CPU utilization on a server logged every 5 minutes for 24 hours "
        "(288 readings). During off-peak hours, utilization is ~15%. During peak, "
        "it jumps to ~65%. The transition times vary by day. Model the daily "
        "usage pattern as states.",
        "family": "changepoint",
        "complexity": "medium",
    },
    {
        "prompt": "River flow measurements (cubic meters/sec) daily for 180 days. Flow is low "
        "during dry season (~50 m3/s), transitions to high during wet season "
        "(~200 m3/s). The transition is gradual over about 2 weeks. Model the "
        "seasonal shift.",
        "family": "changepoint",
        "complexity": "medium",
    },
    {
        "prompt": "A baby's sleep patterns over 100 nights. Time asleep shifts from ~16 hours "
        "(newborn) to ~12 hours (6 months) at some developmental transition point. "
        "Night-to-night variation is about 1.5 hours. Model the developmental shift.",
        "family": "changepoint",
        "complexity": "simple",
    },
    # --- DIRICHLET PROCESS / CRP (10) ---
    {
        "prompt": "Gene expression levels for 200 cells. Cells belong to an unknown number of "
        "types. Each type has a characteristic expression level. We observe noisy "
        "expression measurements. Cluster the cells without specifying K.",
        "family": "dp",
        "complexity": "medium",
    },
    {
        "prompt": "Customer purchase amounts from 300 transactions. Customers fall into natural "
        "spending groups (budget, mid-range, premium), but the number of groups and "
        "their boundaries are unknown. Each group's spending clusters around a "
        "typical amount with variation.",
        "family": "dp",
        "complexity": "medium",
    },
    {
        "prompt": "150 astronomical objects with measured brightness. Objects belong to different "
        "classes (star types), but the number of classes is unknown. Each class has "
        "a characteristic brightness with noise. Cluster by brightness.",
        "family": "dp",
        "complexity": "medium",
    },
    {
        "prompt": "Arrival times of 200 earthquakes in a region over 10 years. Earthquakes "
        "cluster around unknown fault lines. The number of active faults is unknown. "
        "Each fault produces events with a characteristic temporal signature.",
        "family": "dp",
        "complexity": "complex",
    },
    {
        "prompt": "Document word counts: 100 documents, each with a single topic keyword. "
        "Topics group naturally, but the number of distinct topics is unknown. "
        "Each topic has a characteristic word count range. Cluster documents by topic.",
        "family": "dp",
        "complexity": "simple",
    },
    {
        "prompt": "Heights of 250 adult penguins from mixed species. The number of species "
        "present is unknown. Each species has a characteristic height distribution. "
        "Heights are measured with 1cm precision.",
        "family": "dp",
        "complexity": "simple",
    },
    {
        "prompt": "Music tracks with 180 BPM (beats per minute) measurements. Tracks cluster "
        "by genre (ballad ~70, pop ~120, techno ~140, etc.), but the number of genres "
        "and their typical BPMs are unknown. Discover the clusters.",
        "family": "dp",
        "complexity": "medium",
    },
    {
        "prompt": "Reaction times from 200 trials in a cognitive experiment. Some trials are "
        "typical responses, others are guesses, and possibly there are attentional "
        "lapses too. Each response mode has a different typical reaction time. The "
        "number of modes is unknown.",
        "family": "dp",
        "complexity": "complex",
    },
    {
        "prompt": "Weights of 120 fish caught in a lake. Multiple species are present, each "
        "with its own size distribution. The number of species is unknown. Each fish's "
        "weight is noisy within its species. Group the fish by species.",
        "family": "dp",
        "complexity": "simple",
    },
    {
        "prompt": "Duration of phone calls from 300 call records. Calls naturally group by type "
        "(quick check-ins, regular conversations, long support calls). The number of "
        "call types and their typical durations are unknown. Cluster the calls.",
        "family": "dp",
        "complexity": "medium",
    },
    # --- INDIAN BUFFET PROCESS / SPARSE FEATURES (12) ---
    {
        "prompt": "A survey of 100 restaurants rated on an unknown number of latent qualities "
        "(e.g., ambiance, service speed, food quality). Each restaurant exhibits a "
        "sparse subset of these qualities. Overall rating (1-10) is the sum of quality "
        "contributions. Discover the latent quality dimensions from ratings and partial "
        "attribute flags.",
        "family": "ibp",
        "complexity": "medium",
    },
    {
        "prompt": "80 movies described by an unknown set of latent themes "
        "(action, romance, comedy, "
        "etc.). Each movie activates a sparse subset of themes. Audience rating (0-5) "
        "depends on which themes are active and their appeal. Discover the latent "
        "theme structure from ratings alone.",
        "family": "ibp",
        "complexity": "medium",
    },
    {
        "prompt": "150 patients tested for symptom presence across 20 observed symptoms. An "
        "unknown number of underlying conditions each cause a sparse subset of symptoms. "
        "Each patient has one or more conditions. Identify the latent conditions from "
        "binary symptom data.",
        "family": "ibp",
        "complexity": "complex",
    },
    {
        "prompt": "60 photographs tagged with visual attributes (bright, dark, indoor, outdoor, "
        "etc.). Photos share latent visual styles, with each photo exhibiting a sparse "
        "combination. 12 observed binary tags per photo. Discover the latent style "
        "dimensions.",
        "family": "ibp",
        "complexity": "medium",
    },
    {
        "prompt": "90 students each take 10 quizzes. Each quiz tests an unknown number of "
        "latent skills. Each student masters a sparse subset of skills. Quiz score "
        "(0-100) depends on which relevant skills the student has. Discover the "
        "skill structure.",
        "family": "ibp",
        "complexity": "complex",
    },
    {
        "prompt": "120 products in a store categorized by customer purchase patterns. Each "
        "product has unknown latent appeal factors (brand loyalty, price sensitivity, "
        "seasonal demand). Each product activates a sparse set of these factors. "
        "Weekly sales (10-500 units) depend on active factors. Find the factors.",
        "family": "ibp",
        "complexity": "medium",
    },
    {
        "prompt": "70 rock samples analyzed for mineral composition. Each sample contains a "
        "sparse subset of unknown mineral phases. Measurements include density "
        "(2.0-4.0 g/cm3) and hardness (1-10). The number of distinct mineral phases "
        "is unknown. Discover the phases from physical properties.",
        "family": "ibp",
        "complexity": "medium",
    },
    {
        "prompt": "200 social media posts each with engagement score (0-1000). Posts share "
        "latent viral factors (emotional appeal, novelty, controversy, etc.). Each "
        "post activates a sparse subset of these factors. Engagement depends on the "
        "combination of active factors. Discover the factor structure.",
        "family": "ibp",
        "complexity": "complex",
    },
    {
        "prompt": "50 protein sequences characterized by 8 binary structural features (alpha "
        "helix, beta sheet, disulfide bonds, etc.). An unknown number of structural "
        "motifs each correspond to a sparse combination of features. Each protein "
        "realizes a few motifs. Infer the motif dictionary.",
        "family": "ibp",
        "complexity": "complex",
    },
    {
        "prompt": "110 wines rated by 5 critics. Each wine has unknown latent flavor profiles "
        "(fruity, oaky, tannic, etc.) that each critic weights differently. Each wine "
        "expresses a sparse subset of profiles. Ratings are 50-100. Discover the "
        "flavor profiles from the rating matrix.",
        "family": "ibp",
        "complexity": "medium",
    },
    {
        "prompt": "85 network traffic sessions with 6 measured features (packet size, duration, "
        "port count, etc.). Sessions are driven by a sparse set of latent behavior "
        "types (browsing, streaming, malware, etc.). The number of types is unknown. "
        "Discover the behavior types from traffic features.",
        "family": "ibp",
        "complexity": "complex",
    },
    {
        "prompt": "130 archaeological artifacts with 5 measured properties (weight, length, "
        "thickness, decoration count, material hardness). Artifacts share latent "
        "cultural traditions, each contributing a sparse set of characteristic "
        "features. The number of traditions is unknown. Discover them.",
        "family": "ibp",
        "complexity": "medium",
    },
    # --- LOG-GAUSSIAN COX PROCESS (8) ---
    {
        "prompt": "Locations of 200 trees in a 1km x 1km forest plot. Tree density varies "
        "smoothly across the plot, with higher density near water sources. Model the "
        "spatially varying intensity of tree occurrence from the point pattern.",
        "family": "lgcp",
        "complexity": "complex",
    },
    {
        "prompt": "Timestamps of 150 earthquakes in a seismic zone over 20 years. The rate of "
        "earthquakes varies smoothly over time, with periods of heightened activity. "
        "Model the time-varying event intensity from occurrence times.",
        "family": "lgcp",
        "complexity": "medium",
    },
    {
        "prompt": "Locations of 300 crime incidents in a city district (2D coordinates, 0-10km "
        "range). Crime intensity varies smoothly across space, with hotspots near "
        "transit stations. Predict the intensity surface from incident locations.",
        "family": "lgcp",
        "complexity": "complex",
    },
    {
        "prompt": "Arrival times of 100 emergency calls over 48 hours. Call rate varies "
        "smoothly: higher during evening hours, lower at night. Model the "
        "time-varying rate from arrival timestamps.",
        "family": "lgcp",
        "complexity": "medium",
    },
    {
        "prompt": "Positions of 180 bird nests in a 500m x 500m habitat. Nesting density "
        "varies smoothly with vegetation cover (correlated with position). Model "
        "the spatially varying nesting intensity.",
        "family": "lgcp",
        "complexity": "medium",
    },
    {
        "prompt": "Timestamps of 250 website purchases over 30 days. Purchase rate varies "
        "smoothly: peaks on weekdays, dips on weekends, with a gradual uptrend. "
        "Model the smooth intensity function from event times.",
        "family": "lgcp",
        "complexity": "medium",
    },
    {
        "prompt": "Locations of 120 disease cases in a 20km x 20km region. Infection risk "
        "varies smoothly across space, influenced by population density and "
        "environmental factors. Model the spatial intensity surface.",
        "family": "lgcp",
        "complexity": "complex",
    },
    {
        "prompt": "Positions of 90 lightning strikes in a 100km x 100km area during a storm. "
        "Strike intensity varies smoothly, concentrated along a storm front that "
        "moved across the region. Model the spatial pattern.",
        "family": "lgcp",
        "complexity": "medium",
    },
    # --- HIERARCHICAL MODELS (12) ---
    {
        "prompt": "Test scores from 8 schools, with 20-40 students per school. Each school has "
        "a different average score, but schools in the same district may be similar. "
        "Student scores range 200-800. Estimate each school's true average while "
        "sharing information across schools.",
        "family": "hierarchical",
        "complexity": "medium",
    },
    {
        "prompt": "Radon measurements in 50 houses across 10 counties. Each county has a "
        "characteristic radon level, but individual houses vary. Radon levels range "
        "0.5-20 pCi/L. Basement measurements tend to be higher than first-floor. "
        "Estimate county-level radon while borrowing strength.",
        "family": "hierarchical",
        "complexity": "complex",
    },
    {
        "prompt": "Clinical trial results from 12 hospitals testing a drug. Each hospital "
        "reports a treatment effect (difference in recovery days: -5 to +2). "
        "Hospital sample sizes vary from 20 to 100. Estimate the overall drug "
        "effect while allowing hospital-level variation.",
        "family": "hierarchical",
        "complexity": "medium",
    },
    {
        "prompt": "Batting averages of 40 baseball players early in the season (first 50 "
        "at-bats each). True skill varies by player. Early-season averages are "
        "noisy (range 0.15-0.40). Estimate true skill by partially pooling "
        "individual stats toward the league mean.",
        "family": "hierarchical",
        "complexity": "medium",
    },
    {
        "prompt": "Reaction times from 25 participants, each completing 30 trials. Each "
        "person has a characteristic baseline speed (200-500ms) with trial-to-trial "
        "noise (~50ms). Estimate each person's true baseline while sharing "
        "information across the group.",
        "family": "hierarchical",
        "complexity": "simple",
    },
    {
        "prompt": "Customer satisfaction ratings (1-5 stars) for 15 products across 6 "
        "categories. Each category has a baseline quality level. Products within a "
        "category vary around the baseline. 20-50 ratings per product. Estimate "
        "true product quality with category-level pooling.",
        "family": "hierarchical",
        "complexity": "medium",
    },
    {
        "prompt": "Crop yields (tons/hectare) from 30 farms in 5 regions over 3 years. "
        "Regions differ in average yield (2-8 t/ha). Farms within a region vary. "
        "Year-to-year weather effects are shared across farms. Estimate farm-level "
        "productivity with regional pooling.",
        "family": "hierarchical",
        "complexity": "complex",
    },
    {
        "prompt": "Blood pressure readings from 20 patients, each measured 5 times over 6 "
        "months. Each patient has a true baseline (110-160 mmHg) with measurement "
        "noise (~10 mmHg). Patients share a population distribution. Estimate each "
        "patient's true level.",
        "family": "hierarchical",
        "complexity": "simple",
    },
    {
        "prompt": "Error rates of 18 machine learning models on 8 benchmark tasks. Each model "
        "has overall quality, and each task has inherent difficulty. Error rates "
        "range 0.01-0.50. Estimate model and task effects simultaneously with "
        "partial pooling.",
        "family": "hierarchical",
        "complexity": "complex",
    },
    {
        "prompt": "Speed measurements of 35 delivery trucks across 7 routes. Each route has "
        "a characteristic average speed (20-60 km/h), and each truck has a driver "
        "tendency (fast or slow). 10 trips per truck-route pair. Estimate route "
        "and truck effects jointly.",
        "family": "hierarchical",
        "complexity": "complex",
    },
    {
        "prompt": "Reading comprehension scores from 200 students across 10 classrooms. "
        "Each classroom has a different teaching approach and average score. Student "
        "scores range 30-100. Classroom sizes vary (15-25 students). Estimate "
        "classroom effects with partial pooling.",
        "family": "hierarchical",
        "complexity": "medium",
    },
    {
        "prompt": "Daily step counts from 15 fitness tracker users over 30 days. Each user "
        "has a baseline activity level (3000-15000 steps/day). Day-to-day variation "
        "is about 2000 steps. Users share a population pattern. Estimate each "
        "user's true activity level.",
        "family": "hierarchical",
        "complexity": "simple",
    },
    # --- POINT PROCESSES (10) ---
    {
        "prompt": "Timestamps of 80 customer arrivals at a store over 8 hours. Arrival rate "
        "increases during lunch hour and evening rush. Times are in minutes. Model "
        "the time-varying arrival intensity.",
        "family": "point_process",
        "complexity": "medium",
    },
    {
        "prompt": "Times of 120 retweets of a viral post over 24 hours. Early retweets "
        "trigger more retweets (self-exciting pattern). Rate decays over time but "
        "spikes after each burst. Model the self-exciting dynamics.",
        "family": "point_process",
        "complexity": "complex",
    },
    {
        "prompt": "Failure times of 50 industrial components over 1000 operating hours. "
        "Failure rate increases with age (wear-out pattern). Some components fail "
        "early (infant mortality). Model the failure intensity from event times.",
        "family": "point_process",
        "complexity": "medium",
    },
    {
        "prompt": "Timestamps of 200 taxi pickups in a 12-hour shift. Pickup rate varies: "
        "high around airports and train stations, low in residential areas. Events "
        "occur at specific locations. Model the spatiotemporal event pattern.",
        "family": "point_process",
        "complexity": "complex",
    },
    {
        "prompt": "Times of 60 volcanic eruptions from a single volcano over 500 years. "
        "Eruptions come in clusters: active periods followed by quiescence. The "
        "current rate depends on recent activity. Model the temporal clustering.",
        "family": "point_process",
        "complexity": "complex",
    },
    {
        "prompt": "Arrival times of 150 emails over a work week (5 days, 8 hours each). "
        "Email rate is higher in the morning and after lunch. Different days "
        "have different patterns. Model the intensity from timestamps.",
        "family": "point_process",
        "complexity": "medium",
    },
    {
        "prompt": "Times between 90 neural spike events recorded from a single neuron "
        "over 60 seconds. The neuron fires in bursts (clusters of rapid spikes "
        "separated by quiet periods). Inter-spike intervals range 5-500ms. "
        "Model the bursty firing pattern.",
        "family": "point_process",
        "complexity": "complex",
    },
    {
        "prompt": "Timestamps of 100 insurance claims over 2 years. Claims are not evenly "
        "spread: more occur in winter months and after natural disasters. "
        "Model the seasonal and event-driven claim intensity.",
        "family": "point_process",
        "complexity": "medium",
    },
    {
        "prompt": "Times of 70 goals scored across 40 football matches. Goal rate is higher "
        "early in each half (first 15 minutes) and lower in the middle. Goals may "
        "cluster in time (momentum effects). Model the within-match intensity.",
        "family": "point_process",
        "complexity": "medium",
    },
    {
        "prompt": "Occurrence times of 110 traffic accidents on a highway over 1 year. "
        "Accident rate varies by time of day (rush hours) and weather (rain increases "
        "risk). Daily pattern repeats but with random variation. Model the "
        "time-varying accident intensity.",
        "family": "point_process",
        "complexity": "medium",
    },
]


def generate_one(
    prompt_text: str,
    model: str,
    max_repairs: int = 3,
    temperature: float = 0.4,
) -> dict:
    resp = litellm_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        temperature=temperature,
        drop_params=True,
    )
    raw_completion = resp.choices[0].message.content or ""

    code = LAZYPPL.extract_code(raw_completion)
    if code is None:
        return {
            "code": None,
            "completion": raw_completion,
            "think": None,
            "repairs": 0,
            "typecheck_ok": False,
            "error": "no code block",
        }

    think_match = re.search(r"<think>(.*?)</think>", raw_completion, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else None

    result = LAZYPPL.validate_runtime(code)
    repairs = 0

    while not result.ok and repairs < max_repairs:
        repairs += 1
        sys_prompt, user_msg = LAZYPPL.repair_system_prompt(
            prompt_text=prompt_text,
            broken_code=code,
            error_trace=result.message,
        )
        repair_resp = litellm_completion(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            drop_params=True,
        )
        new_code = LAZYPPL.extract_code(repair_resp.choices[0].message.content)
        if new_code is not None:
            code = new_code
            raw_completion = LAZYPPL.replace_code(raw_completion, code)
        result = LAZYPPL.validate_runtime(code)

    completion_out = re.sub(r"<think>.*?</think>\s*", "", raw_completion, flags=re.DOTALL).strip()

    return {
        "code": code,
        "completion": completion_out,
        "think": think_text,
        "repairs": repairs,
        "typecheck_ok": result.ok,
        "error": None if result.ok else result.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate native LazyPPL records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = NATIVE_PROMPTS[: args.limit] if args.limit > 0 else NATIVE_PROMPTS
    total = len(prompts)
    ok_count = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(prompts):
            print(f"[{idx + 1}/{total}] {entry['family']}...", end=" ", flush=True)

            result = generate_one(
                prompt_text=entry["prompt"],
                model=args.model,
                max_repairs=args.max_repairs,
                temperature=args.temperature,
            )

            rec = {
                "prompt": entry["prompt"],
                "completion": result["completion"],
                "metadata": {
                    "model_type": "inferential",
                    "domain": "native_lazyppl",
                    "complexity": entry["complexity"],
                    "key_concepts": [entry["family"]],
                    "has_observed_data": True,
                    "lazyppl_version": "lazyppl-1.0",
                    "source": "native_lazyppl",
                    "family": entry["family"],
                    "typecheck_ok": result["typecheck_ok"],
                    "repairs": result["repairs"],
                },
                "complexity": entry["complexity"],
            }
            if result["think"]:
                rec["metadata"]["think"] = result["think"]
            if result["error"]:
                rec["metadata"]["error"] = result["error"]

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            status = "OK" if result["typecheck_ok"] else "FAIL"
            print(f"{status}")

            if result["typecheck_ok"]:
                ok_count += 1

            if args.delay > 0 and idx < total - 1:
                time.sleep(args.delay)

    print(f"\nDone. {ok_count}/{total} typecheck OK.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

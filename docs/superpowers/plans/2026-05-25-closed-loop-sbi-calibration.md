# Closed-Loop SBI Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate SBI (Single-Round SNPE) calibration into the closed-loop workflow and remove all electrolyte dryout profile dependencies.

**Architecture:** We will define parameter priors, extract flat feature vectors (SOH, DCIR, ICA peak V, peak H, area) across checkpoints, train a Masked Autoregressive Flow (MAF) posterior estimator using PyTorch/SBI, and estimate parameters on mock/real data.

**Tech Stack:** PyTorch, sbi, PyBaMM, pandas, scikit-learn

---

### Task 1: Requirements and Configuration Updates

**Files:**
- Modify: `requirements.txt`
- Delete: `03_simulation/closed_loop/data/electrolyte_dryout_profile.csv`

- [x] **Step 1: Write requirements.txt edits**
- [x] **Step 2: Commit Task 1**

---

### Task 2: Refactor Coupled Degradation Simulator to remove Dryout Profile

**Files:**
- Modify: `03_simulation/closed_loop/src/degradation/coupled_degradation_model.py`

- [x] **Step 1: Remove dryout profile loading and updates**
- [x] **Step 2: Update CoupledDegradationSimulator.run() signature**
- [x] **Step 3: Commit Task 2**

---

### Task 3: Refactor RPT Calibrator to remove Dryout Profile

**Files:**
- Modify: `03_simulation/closed_loop/src/calibration/rpt_calibration.py`
- Modify: `03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py`

- [x] **Step 1: Remove dryout profile in rpt_calibration.py**
- [x] **Step 2: Remove dryout_scale in vector_to_updates**
- [x] **Step 3: Update closed_loop_pipeline.py parser and calls**
- [x] **Step 4: Commit Task 3**

---

### Task 4: Implement SBI (SNPE) Calibration in rpt_calibration.py

**Files:**
- Modify: `03_simulation/closed_loop/src/calibration/rpt_calibration.py`

- [x] **Step 1: Write helper to extract concatenated feature vectors**
- [x] **Step 2: Implement calibrate_sbi method in RPTCalibrator**
- [x] **Step 3: Commit Task 4**

---

### Task 5: Integrate SBI into closed_loop_pipeline.py

**Files:**
- Modify: `03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py`

- [x] **Step 1: Update pipeline command line parser and main invocation**
- [x] **Step 2: Commit Task 5**

---

### Task 6: Create Mock Data Templates for Closed-Loop

**Files:**
- Create: `03_simulation/closed_loop/data/real_rpt_summary.csv`
- Create: `03_simulation/closed_loop/data/real_rpt_low_rate_trace.csv`
- Create: `03_simulation/closed_loop/data/real_cycle_timeseries.csv`
- Create: `03_simulation/closed_loop/data/cycle_protocols.csv`

- [x] **Step 1: Write template data**
- [x] **Step 2: Commit Task 6**

---

### Task 7: Unit Testing for SBI Closed-Loop Calibration

**Files:**
- Create: `03_simulation/closed_loop/tests/test_sbi_calibration.py`

- [x] **Step 1: Write unit tests**
- [x] **Step 2: Commit Task 7**

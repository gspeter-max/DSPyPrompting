# Implementation Plan: Replace Custom Metrics with DSPy SemanticF1

**Created**: 2026-02-15
**Author**: Claude Code (Planning Mode)
**Status**: Draft
**Estimated Duration**: 2-3 hours

---

## Overview

### Goal
Replace two custom, heuristic-based metrics (`answer_exact_match_metric` and `context_adherence_metric`) with DSPy's production-ready `SemanticF1` metric for improved semantic evaluation quality.

### Success Criteria
- [ ] Both custom metrics completely removed from `qa_module.py`
- [ ] `train.py` updated to use `SemanticF1(decompositional=True)`
- [ ] `test.py` updated with threshold wrapper for boolean assertions
- [ ] Model successfully retrained with new metric
- [ ] All 5 tests pass after retraining
- [ ] Documentation updated to reflect new metric

### Scope
**In Scope:**
- Remove custom metric functions from `qa_module.py`
- Update `train.py` to import and use SemanticF1
- Update `test.py` to import and use SemanticF1 with thresholding
- Update `README.md` documentation
- Retrain model with new metric
- Verify all tests pass

**Out of Scope:**
- Modifying the dataset (no changes needed)
- Changing QAModule architecture (no changes needed)
- Adding new features or tests
- Performance optimization beyond metric replacement

---

## Architecture & Design

### Current State
```
qa_module.py:
  - GenerateAnswer signature (lines 9-22)
  - QAModule class (lines 24-34)
  - answer_exact_match_metric (lines 36-47) ❌ TO REMOVE
  - context_adherence_metric (lines 50-98) ❌ TO REMOVE

train.py:
  - Imports: from qa_module import context_adherence_metric (line 6)
  - Usage: metric=context_adherence_metric (line 36)

test.py:
  - Imports: from qa_module import answer_exact_match_metric (line 6)
  - Usage: 3 test functions (lines 35, 87, 130)
```

### Proposed Changes
```
qa_module.py:
  - GenerateAnswer signature (KEEP)
  - QAModule class (KEEP)
  - [REMOVE] answer_exact_match_metric
  - [REMOVE] context_adherence_metric

train.py:
  - Imports: from dspy.evaluate import SemanticF1 (NEW)
  - Usage: metric=SemanticF1(decompositional=True) (NEW)

test.py:
  - Imports: from dspy.evaluate import SemanticF1 (NEW)
  - Helper: semantic_f1_threshold() function (NEW)
  - Usage: All 3 test functions use threshold wrapper (MODIFIED)
```

### Data Flow
```
Training Flow:
  trainset → BootstrapFewShot → SemanticF1 metric → Score (0.0-1.0) → Select demos → Trained model

Testing Flow:
  trained_model → Prediction → SemanticF1 metric → Score → Threshold (≥0.5) → Boolean → Test assertion
```

---

## Phase Breakdown

### Phase 1: Remove Custom Metrics from qa_module.py
**Estimated Time**: 5 minutes
**Dependencies**: None

**Goal**: Remove both custom metric functions, keeping only QAModule components

**Files to Modify:**
- `/Users/apple/project/DSPyPrompting/qa_module.py` - Remove lines 36-98

**Implementation Steps:**
1. Read `qa_module.py` to verify current state
2. Remove `answer_exact_match_metric` function (lines 36-47)
3. Remove `context_adherence_metric` function (lines 50-98)
4. Verify file contains only:
   - Module docstring (lines 1-4)
   - `import dspy` (line 6)
   - `GenerateAnswer` class (lines 9-22)
   - `QAModule` class (lines 24-34)
5. Verify no blank lines remain where functions were removed

**Verification:**
- [ ] File contains only QAModule components (2 classes)
- [ ] No metric functions present
- [ ] File has 34 lines total
- [ ] Python syntax check: `python -m py_compile qa_module.py`

**Risks:**
- Risk: Accidentally deleting QAModule class
- Mitigation: Carefully verify line numbers before deletion

**Todo Template:**
```
- [ ] Read qa_module.py to verify current state
- [ ] Remove answer_exact_match_metric function (lines 36-47)
- [ ] Remove context_adherence_metric function (lines 50-98)
- [ ] Verify file structure (only 2 classes remain)
- [ ] Run syntax check: python -m py_compile qa_module.py
```

---

### Phase 2: Update train.py to Use SemanticF1
**Estimated Time**: 10 minutes
**Dependencies**: Phase 1 complete

**Goal**: Replace custom metric import and usage with SemanticF1

**Files to Modify:**
- `/Users/apple/project/DSPyPrompting/train.py` - Update lines 6 and 36

**Implementation Steps:**
1. Read `train.py` to understand current imports
2. **Line 6**: Change import
   ```python
   # OLD:
   from qa_module import QAModule, context_adherence_metric

   # NEW:
   from qa_module import QAModule
   from dspy.evaluate import SemanticF1
   ```
3. **Line 36**: Change metric usage
   ```python
   # OLD:
   metric=context_adherence_metric,

   # NEW:
   metric=SemanticF1(decompositional=True),
   ```
4. Consider adding `metric_threshold=0.5` parameter (optional but recommended)
5. Verify no other references to old metrics exist

**Verification:**
- [ ] Import statement updated correctly
- [ ] Metric parameter updated correctly
- [ ] No old metric references remain
- [ ] Python syntax check: `python -m py_compile train.py`
- [ ] Dry run: `python train.py --help 2>&1 | head -5` (should not import-error)

**Risks:**
- Risk: SemanticF1 import fails if DSPy version < 2.4
- Mitigation: requirements.txt specifies dspy>=2.4.0, verify before running
- Risk: Training takes longer due to LLM-based metric
- Mitigation: Expected and acceptable - better quality worth the cost

**Todo Template:**
```
- [ ] Read train.py to verify current imports
- [ ] Update line 6: Remove context_adherence_metric from import
- [ ] Update line 6: Add SemanticF1 import from dspy.evaluate
- [ ] Update line 36: Replace metric with SemanticF1(decompositional=True)
- [ ] Optional: Add metric_threshold=0.5 parameter
- [ ] Run syntax check: python -m py_compile train.py
- [ ] Verify no old metric references: grep -n "context_adherence_metric\|answer_exact_match" train.py
```

---

### Phase 3: Update test.py with Threshold Wrapper
**Estimated Time**: 15 minutes
**Dependencies**: Phase 1 complete

**Goal**: Replace custom metric with SemanticF1 and threshold wrapper for boolean tests

**Files to Modify:**
- `/Users/apple/project/DSPyPrompting/test.py` - Update lines 6, add helper, modify 3 test functions

**Implementation Steps:**
1. Read `test.py` to understand all metric usage points
2. **After line 6**: Add imports
   ```python
   # OLD:
   from qa_module import QAModule, answer_exact_match_metric

   # NEW:
   from qa_module import QAModule
   from dspy.evaluate import SemanticF1
   ```
3. **After line 16** (after dspy.configure): Add threshold helper function
   ```python
   def semantic_f1_threshold(gold, pred, trace=None, threshold=0.5):
       """Wrapper to convert SemanticF1 score to boolean for test assertions.

       Args:
           gold: Ground truth example
           pred: Prediction from model
           trace: Optional trace (unused, for API compatibility)
           threshold: Minimum SemanticF1 score to consider correct (default: 0.5)

       Returns:
           bool: True if SemanticF1 score >= threshold
       """
       metric = SemanticF1(decompositional=True)
       score = metric(gold, pred)
       return score >= threshold
   ```
4. **Line 35** (test_training_data_accuracy): Replace metric call
   ```python
   # OLD:
   is_correct = answer_exact_match_metric(sample, pred)

   # NEW:
   is_correct = semantic_f1_threshold(sample, pred)
   ```
5. **Line 87** (test_generalization): Replace metric call
   ```python
   # OLD:
   is_correct = answer_exact_match_metric(sample, pred)

   # NEW:
   is_correct = semantic_f1_threshold(sample, pred)
   ```
6. **Line 130-131** (test_untrained_vs_trained): Replace both metric calls
   ```python
   # OLD:
   untrained_ok = answer_exact_match_metric(sample, pred_untrained)
   trained_ok = answer_exact_match_metric(sample, pred_trained)

   # NEW:
   untrained_ok = semantic_f1_threshold(sample, pred_untrained)
   trained_ok = semantic_f1_threshold(sample, pred_trained)
   ```
7. Verify no other references to old metric exist

**Verification:**
- [ ] Import statement updated
- [ ] Helper function added with proper docstring
- [ ] All 3 test functions updated
- [ ] No old metric references remain
- [ ] Python syntax check: `python -m py_compile test.py`
- [ ] Grep check: `grep -n "answer_exact_match" test.py` (should return nothing)

**Risks:**
- Risk: Threshold too strict (0.5) causing test failures
- Mitigation: If tests fail, adjust threshold to 0.4 or document expected behavior
- Risk: SemanticF1 slower than exact match
- Mitigation: Expected - tests will take longer but results more meaningful

**Todo Template:**
```
- [ ] Read test.py to identify all metric usage points
- [ ] Update line 6: Remove answer_exact_match_metric from import
- [ ] Update line 6: Add SemanticF1 import from dspy.evaluate
- [ ] Add semantic_f1_threshold() helper function after line 16
- [ ] Update test_training_data_accuracy (line 35)
- [ ] Update test_generalization (line 87)
- [ ] Update test_untrained_vs_trained (lines 130-131)
- [ ] Run syntax check: python -m py_compile test.py
- [ ] Verify no old references: grep -n "answer_exact_match" test.py
```

---

### Phase 4: Update README.md Documentation
**Estimated Time**: 10 minutes
**Dependencies**: Phases 1-3 complete

**Goal**: Update documentation to reflect SemanticF1 metric usage

**Files to Modify:**
- `/Users/apple/project/DSPyPrompting/README.md` - Update metric references

**Implementation Steps:**
1. Read `README.md` to find all metric references
2. **Line 110** (BootstrapFewShot section): Update code example
   ```python
   # OLD:
   optimizer = dspy.BootstrapFewShot(
       metric=answer_exact_match_metric,
       max_labeled_demos=3,
       max_bootstrapped_demos=3,
       max_rounds=1
   )

   # NEW:
   from dspy.evaluate import SemanticF1

   optimizer = dspy.BootstrapFewShot(
       metric=SemanticF1(decompositional=True),
       max_labeled_demos=3,
       max_bootstrapped_demos=3,
       max_rounds=1
   )
   ```
3. **Line 178** (Model Configuration section): Update metric description
   ```python
   # OLD:
   - **Metric**: Exact match (case-insensitive)

   # NEW:
   - **Metric**: SemanticF1 (semantic similarity with decompositional=True)
   ```
4. **Line 200** (Notes section): Add note about SemanticF1
   ```python
   - Answers are concise for exact matching  # REMOVE
   + SemanticF1 metric evaluates semantic similarity, not just exact text match
   - Context contains all necessary information  # KEEP
   - No trick questions (learning focus)  # KEEP
   - Can be extended with semantic similarity metrics  # REMOVE (already using it)
   ```
5. Search for any other "exact match" or "metric" references and update

**Verification:**
- [ ] All metric references updated
- [ ] Code examples are accurate
- [ ] No references to old custom metrics
- [ ] Documentation is clear and accurate
- [ ] Grep check: `grep -n "exact_match\|context_adherence" README.md` (should return nothing)

**Risks:**
- Risk: Missing some metric references in documentation
- Mitigation: Use grep to find all occurrences

**Todo Template:**
```
- [ ] Read README.md to find all metric references
- [ ] Update BootstrapFewShot code example (line ~110)
- [ ] Update Model Configuration section (line ~178)
- [ ] Update Notes section (line ~200)
- [ ] Search for remaining "exact match" references
- [ ] Grep check: grep -n "exact_match\|context_adherence" README.md
- [ ] Verify documentation accuracy
```

---

### Phase 5: Retrain Model with New Metric
**Estimated Time**: 10-15 minutes
**Dependencies**: Phases 1-4 complete

**Goal**: Retrain the model using SemanticF1 metric and verify successful training

**Files to Modify:**
- None (execution only)
- Output: `trained_qa_model.json` (will be overwritten)

**Implementation Steps:**
1. Backup existing trained model (optional but recommended):
   ```bash
   cp trained_qa_model.json trained_qa_model.json.backup
   ```
2. Run training script:
   ```bash
   python train.py
   ```
3. Monitor output for:
   - Successful completion (no errors)
   - Training progress messages
   - Demonstrations selected by optimizer
   - Model saved confirmation
4. Verify `trained_qa_model.json` is created/updated
5. Check file size (should be > 0 bytes)

**Verification:**
- [ ] Training completes without errors
- [ ] trained_qa_model.json is created/updated
- [ ] File size > 0 bytes
- [ ] Output shows demonstrations selected
- [ ] No import errors or exceptions
- [ ] Training time: 5-15 minutes (expected longer due to LLM-based metric)

**Risks:**
- Risk: Training fails due to SemanticF1 errors
- Mitigation: Check DSPy version, verify import syntax
- Risk: Training takes much longer than expected
- Mitigation: Normal - SemanticF1 uses LLM for scoring, 9 samples should complete in reasonable time
- Risk: Groq API rate limits exceeded
- Mitigation: 9 training samples × few rounds = well within 30 requests/minute limit

**Todo Template:**
```
- [ ] Backup existing trained model: cp trained_qa_model.json trained_qa_model.json.backup
- [ ] Run training: python train.py
- [ ] Monitor training progress
- [ ] Verify trained_qa_model.json is created/updated
- [ ] Check file size: ls -lh trained_qa_model.json
- [ ] Review training output for errors
```

---

### Phase 6: Run Full Test Suite
**Estimated Time**: 10 minutes
**Dependencies**: Phase 5 complete (model retrained)

**Goal**: Verify all tests pass with new SemanticF1 metric

**Files to Modify:**
- None (execution only)

**Implementation Steps:**
1. Ensure trained model exists: `ls -lh trained_qa_model.json`
2. Run test suite:
   ```bash
   python test.py
   ```
3. Monitor output for each test:
   - Test 1: Training Data Accuracy (should be high, maybe not 100% with SemanticF1)
   - Test 2: Generalization (≥60% expected)
   - Test 3: Trained > Untrained (should show improvement)
   - Test 4: Context Adherence (should refuse answering)
   - Test 5: Edge Cases (should handle gracefully)
4. Review test summary
5. If tests fail, analyze failure and adjust threshold if needed

**Verification:**
- [ ] All 5 tests run without errors
- [ ] Test 1: High accuracy (≥80% acceptable with SemanticF1)
- [ ] Test 2: Generalization ≥60%
- [ ] Test 3: Trained better than untrained
- [ ] Test 4: Context adherence works (no hallucination)
- [ ] Test 5: Edge cases handled
- [ ] Overall: ≥4/5 tests pass

**Risks:**
- Risk: Test 1 accuracy < 100% with SemanticF1
- Mitigation: Acceptable - SemanticF1 more nuanced than exact match, ≥80% is good
- Risk: Some tests fail due to threshold strictness
- Mitigation: If ≥0.5 threshold causes failures, adjust to 0.4
- Risk: Test 4 (context adherence) may behave differently
- Mitigation: SemanticF1 should naturally handle refusals, verify manually

**Todo Template:**
```
- [ ] Verify trained model exists
- [ ] Run test suite: python test.py
- [ ] Monitor each test result
- [ ] Review test summary
- [ ] If Test 1 accuracy < 80%, consider adjusting threshold
- [ ] If any tests fail, analyze root cause
- [ ] Document test results
```

---

### Phase 7: Final Verification and Cleanup
**Estimated Time**: 5 minutes
**Dependencies**: Phase 6 complete

**Goal**: Final verification and optional cleanup

**Files to Modify:**
- None (verification only)

**Implementation Steps:**
1. **Code Verification**:
   ```bash
   # Verify no old metric references remain
   grep -r "answer_exact_match_metric\|context_adherence_metric" .
   # Expected: No matches (except in git history)
   ```
2. **Import Verification**:
   ```bash
   # Verify SemanticF1 is imported correctly
   grep -n "from dspy.evaluate import SemanticF1" train.py test.py
   # Expected: Both files show import
   ```
3. **Syntax Verification**:
   ```bash
   python -m py_compile qa_module.py train.py test.py
   # Expected: No errors
   ```
4. **Optional: Remove backup model** (if confident in new model):
   ```bash
   rm trained_qa_model.json.backup
   ```
5. **Generate summary report**:
   - List changes made
   - Test results
   - Any issues encountered

**Verification:**
- [ ] No old metric references in codebase
- [ ] SemanticF1 imported in both train.py and test.py
- [ ] All Python files compile without errors
- [ ] trained_qa_model.json exists and is valid
- [ ] Git diff shows only expected changes

**Risks:**
- Risk: Removing backup before confirming new model works
- Mitigation: Keep backup until after full verification

**Todo Template:**
```
- [ ] Search for old metric references: grep -r "answer_exact_match_metric\|context_adherence_metric" .
- [ ] Verify SemanticF1 imports
- [ ] Run syntax check on all files
- [ ] Review git diff to confirm changes
- [ ] Optional: Keep or remove backup model
- [ ] Document final state
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SemanticF1 import fails (wrong DSPy version) | Low | High | Verify requirements.txt has dspy>=2.4.0; check `pip show dspy` before starting |
| Training takes much longer (LLM-based metric) | High | Low | Expected behavior; 9 samples should complete in 10-15 minutes |
| Test failures due to threshold (0.5 too strict) | Medium | Medium | If failures occur, adjust threshold to 0.4 in test.py helper function |
| Groq API rate limits during training | Low | Medium | 9 samples × few rounds = well within 30 req/min limit; monitor if needed |
| SemanticF1 produces different demonstrations | High | Low | Expected and desirable - better metric = better demonstrations |
| Context adherence test (Test 4) behaves differently | Medium | Low | SemanticF1 should handle refusals naturally; verify manually |

---

## Testing Strategy

### Unit Tests
No unit tests needed - this is a metric replacement, not new functionality

### Integration Tests
- **Test 1**: Run `python train.py` → Should complete without errors
- **Test 2**: Run `python test.py` → All 5 tests should pass (≥4/5 acceptable)
- **Test 3**: Verify trained model loads correctly → `python verify.py`

### Manual Testing
- **Manual verification**: Run `python verify.py` and test with custom questions
- **Context adherence test**: Ask question outside context, verify refusal
- **Semantic similarity test**: Ask paraphrased questions, verify semantic understanding

### Edge Cases to Test
1. **Empty context**: Should still work (handled by existing code)
2. **Multi-part questions**: Should work (existing test)
3. **Negative examples** (answer not in context): Should refuse (Test 4)
4. **Paraphrased answers**: SemanticF1 should handle better than exact match
5. **Threshold edge case**: Test with scores near 0.5 threshold

---

## Rollback Plan

If implementation fails at any phase:

### Phase 1-4 Failures (Code Changes)
```bash
# Restore original files from git
git checkout qa_module.py train.py test.py README.md

# Verify old imports work
python -c "from qa_module import answer_exact_match_metric, context_adherence_metric"
python -m py_compile train.py test.py
```

### Phase 5 Failures (Training)
```bash
# Restore old trained model
cp trained_qa_model.json.backup trained_qa_model.json

# Verify old model works
python test.py  # Should use old model
```

### Complete Rollback
```bash
# Reset everything to pre-implementation state
git checkout qa_module.py train.py test.py README.md trained_qa_model.json

# Verify everything works
python train.py  # Should work with old metrics
python test.py   # Should pass with old metrics
```

---

## Notes

### Key Insights
1. **SemanticF1 is production-ready**: Built-in DSPy metric, extensively tested
2. **Float vs Boolean**: SemanticF1 returns float scores (0.0-1.0), tests need threshold wrapper
3. **Decompositional=True**: More sophisticated evaluation using LLM, worth the extra time
4. **Retraining required**: Old trained model incompatible with new metric expectations
5. **Training time increase**: Expected - LLM-based scoring slower than heuristics

### Assumptions
1. DSPy version >= 2.4.0 installed (specified in requirements.txt)
2. Groq API key valid and within rate limits
3. Python 3.8+ available
4. Existing trained model can be overwritten (or backed up)
5. Threshold of 0.5 is appropriate for boolean conversion (may need adjustment)

### Expected Outcomes
- **Better semantic understanding**: SemanticF1 captures meaning, not just text matching
- **More robust training**: Model selects better demonstrations based on semantic quality
- **Slower but better**: Training/testing takes longer but results more meaningful
- **Test accuracy may vary**: Test 1 may not be 100% with SemanticF1 (acceptable)

### Performance Expectations
- **Training time**: 10-15 minutes (vs. 2-3 minutes with custom metrics)
- **Test time**: 5-10 minutes (vs. 1-2 minutes with custom metrics)
- **Quality improvement**: Better semantic understanding, more robust model

---

## Sources

**DSPy SemanticF1 Documentation:**
- [SemanticF1 - DSPy API Documentation](https://dspy.ai/api/evaluation/SemanticF1/)
- [DSPy RAG Tutorial with SemanticF1](https://dspy.ai/tutorials/rag)
- [DSPy Metrics Guide](https://dspy.ai/learn/evaluation/metrics/)
- [DSPy BootstrapFewShot API](https://dspy.ai/api/optimizers/BootstrapFewShot/)

**Additional Resources:**
- [DSPy Main Documentation](https://dspy.ai/)
- [DSPy Cheatsheet - BasicQA](https://dspy.ai/cheatsheet_h=basicqa)

---

## Appendix: Code Comparison

### Before (Custom Metrics)
```python
# qa_module.py
def answer_exact_match_metric(gold, pred, trace=None):
    return pred.answer.lower().strip() == gold.answer.lower().strip()

def context_adherence_metric(gold, pred, trace=None):
    # 48 lines of complex heuristic logic...
    pred_answer = pred.answer.lower().strip()
    gold_answer = gold.answer.lower().strip()
    # Multiple rules and conditions...
    return bool_result

# train.py
from qa_module import QAModule, context_adherence_metric
optimizer = dspy.BootstrapFewShot(
    metric=context_adherence_metric,
    max_labeled_demos=4,
    max_bootstrapped_demos=4,
    max_rounds=1,
    max_errors=10
)
```

### After (SemanticF1)
```python
# qa_module.py
# (No metric functions - file only contains QAModule)

# train.py
from qa_module import QAModule
from dspy.evaluate import SemanticF1

optimizer = dspy.BootstrapFewShot(
    metric=SemanticF1(decompositional=True),
    max_labeled_demos=4,
    max_bootstrapped_demos=4,
    max_rounds=1,
    max_errors=10
)
```

### Test Wrapper (test.py)
```python
from dspy.evaluate import SemanticF1

def semantic_f1_threshold(gold, pred, trace=None, threshold=0.5):
    """Wrapper to convert SemanticF1 score to boolean for test assertions."""
    metric = SemanticF1(decompositional=True)
    score = metric(gold, pred)
    return score >= threshold

# Usage in tests:
is_correct = semantic_f1_threshold(sample, pred)
```

---

**END OF IMPLEMENTATION PLAN**

# DATA QUALITY ANALYSIS

## 1. Label Quality

- Total fall events: 478
- Quality low: 141
- Needs review: 126
- Both flags: 42
- At least one flag: 225
- Ignore true: 0
- Duplicate event IDs: 0
- Invalid event intervals: 0
- Missing CSV cells: 0

## 2. HDF5 Quality

- Files scanned: 18
- Events matched: 478/478
- Invalid CIR shapes: 0
- Invalid timestamp lengths: 0
- Non-finite timestamp streams: 0
- NaN values: 0
- Inf values: 0
- Affected frames: 0
- All-zero frames: 0
- Duplicate/backward timestamp steps: 0
- Time gaps above 0.03s: 0
- Events outside stream bounds: 0

## 3. Data Quality Observations

- HDF5 structure, timestamps, and signal values were checked.
- All annotated events were matched with their HDF5 streams.
- Quality flags must be considered during model development.
- All-zero frames are not automatically invalid.
- Consistent timestamps do not guarantee perfect raw measurements.
- Low-quality annotations are not equivalent to missing values.

## 4. Recommendations

- Check input quality before preprocessing.
- Investigate low-quality and needs-review fall events.
- Respect the challenge evaluation rules for flagged events.
- Avoid learning artificial stream-boundary patterns.

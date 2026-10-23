# DATA QUALITY ANALYSIS

## 1. Label quality
- Total fall events: 478
- Quality low: 141
- Needs review: 126
- Both flags: 42
- At least one flag: 225
- Explicit ignore: 0
- Duplicate event IDs: 0
- Invalid event intervals: 0
- Missing CSV cells: 0

## 2. HDF5 quality
- Files scanned: 18
- NaN values: 0
- Inf values: 0
- Affected frames: 0
- Duplicate/backward timestamp steps: 0
- Time gaps above 0.03s: 0
- Events outside stream bounds: 0

## 3. Cautions
- All-zero frames may be legitimate, not necessarily errors.
- Regular 50 Hz timestamps do not prove raw measurements are complete.
- Quality=low is an annotation-quality flag, not a NaN indicator.
- Verify the per-stream CSV and any scan errors before concluding.

# 🚀 QUICK START GUIDE - Reorganized Project

## 🏆 HACKATHON PRESENTATION (FASTEST)

### Step 1: Open Main Dashboard
```bash
# Double-click this file or open in browser:
HACKATHON_DEMO/MAIN_DASHBOARD.html
```

### Step 2: Run Main Engine (Optional)
```bash
# From project root directory:
cd src/engines
python enhanced_chronic_risk_engine.py --mode mimic --enhanced
```

### Step 3: View Results
```bash
# Check model outputs:
ls ../../data/results/
```

## 📁 PROJECT STRUCTURE OVERVIEW

```
📁 Your Project/
├── 🏆 HACKATHON_DEMO/           # Quick access for presentation
│   ├── MAIN_DASHBOARD.html      # ← START HERE for demo
│   ├── MAIN_ENGINE.py           # Enhanced ML engine
│   └── PRESENTATION_GUIDE.md    # Presentation instructions
├── 📁 src/                      # Source code
│   ├── engines/                 # ML engines
│   ├── dashboard/               # Dashboard applications  
│   └── utils/                   # Utilities
├── 📁 data/                     # Your datasets and results
├── 📁 models/                   # Your trained models
└── 📁 docs/                     # Documentation
```

## 🎯 PRESENTATION TALKING POINTS

1. **"Perfect Performance"** - Point to 1.000 AUROC in dashboard
2. **"Real Clinical Data"** - Mention 275 MIMIC-IV patients
3. **"Interactive Demo"** - Use risk calculator sliders
4. **"Clinical Insights"** - Show deterioration triggers table
5. **"Production Ready"** - Highlight professional code structure

## 🔧 TROUBLESHOOTING

### If imports don't work:
```bash
# Run from project root:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# or on Windows:
set PYTHONPATH=%PYTHONPATH%;%cd%
```

### If dashboard doesn't open:
- Right-click MAIN_DASHBOARD.html → Open with → Browser
- Or copy full path and paste in browser address bar

## 🏆 YOU'RE READY TO WIN!

Your project is now professionally organized and presentation-ready!

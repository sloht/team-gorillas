                                                                                                                                          
"""Score named candidate models on tuning, audit, and full validation episodes.                                                           
                                                                                                                                          
Usage (from main-participants-60-40/):                                                                                                    
    python run_candidates.py --data public --out candidates-01                                                                            
    python run_candidates.py --data public --out candidates-02 --candidates adaptive_lags --skip-audit                                    
                                                                                                                                          
Only <data>/train and <data>/validation are read. <data>/test is never opened.                                                            
Selection discipline: choose settings on the TUNING column. Look at AUDIT only                                                            
for final candidates (use --skip-audit while exploring).                                                                                  
"""                                                                                                                                       
import argparse                                                                                                                           
import json                                                                                                                               
import sys                                                                                                                                
from pathlib import Path                                                                                                                  
from time import perf_counter                                                                                                             
                                                                                                                                          
sys.path.insert(0, str(Path(__file__).resolve().parent))                                                                                  
                                                                                                                                          
from slot_forecasting.advanced import AllTicksForecaster                                                                                  
from slot_forecasting.baseline import FrequencyForecaster                                                                                 
from slot_forecasting.experiments import episode_folds, subset_score                                                                      
from slot_forecasting.io import (                                                                                                         
    load_inputs,                                                                                                                          
    read_labels,                                                                                                                          
    read_rows,                                                                                                                            
    read_submission,                                                                                                                      
    read_targets,                                                                                                                         
    require_ids,                                                                                                                          
    write_json,                                                                                                                           
    write_submission,                                                                                                                     
)                                                                                                                                         
from slot_forecasting.rounds import round_metadata                                                                                        
from slot_forecasting.schema import CONDITIONS, ValidationError                                                                           
                                                                                                                                          
                                                                                                                                          
class Namespace:                                                                                                                          
    def __init__(self, **kwargs):                                                                                                         
        self.__dict__.update(kwargs)                                                                                                      
                                                                                                                                          
                                                                                                                                          
def adaptive_lags():                                                                                                                      
    return AllTicksForecaster(                                                                                                            
        "lags",                                                                                                                           
        strength=24,                                                                                                                      
        local_weight=0.40,                                                                                                                
        half_lives=(8, 24, 100),                                                                                                          
        adaptive=True,                                                                                                                    
    )                                                                                                                                     
                                                                                                                                          
                                                                                                                                          
def pooled_recent():                                                                                                                      
    return FrequencyForecaster("pooled_recent")                                                                                           
                                                                                                                                          
                                                                                                                                          
def registry():                                                                                                                           
    """Name -> factory. adaptive_lags is the baseline row."""                                                                             
    candidates = {                                                                                                                        
        "adaptive_lags": adaptive_lags,                                                                                                   
        "pooled_recent": pooled_recent,                                                                                                   
    }                                                                                                                                     
    try:                                                                                                                                  
        from slot_forecasting import advanced2                                                                                            
    except ModuleNotFoundError as error:                                                                                                  
        if error.name != "slot_forecasting.advanced2":                                                                                    
            raise                                                                                                                         
    else:                                                                                                                                 
        for name, factory in advanced2.candidates():                                                                                      
            if name in candidates:                                                                                                        
                raise ValidationError(f"Duplicate candidate name: {name}.")                                                               
            candidates[name] = factory                                                                                                    
    return candidates                                                                                                                     
                                                                                                                                          
                                                                                                                                          
def _is_number(value):                                                                                                                    
    return isinstance(value, (int, float)) and not isinstance(value, bool)                                                                
                                                                                                                                          
                                                                                                                                          
def _ratio_of(value):                                                                                                                     
    if _is_number(value):                                                                                                                 
        return float(value)                                                                                                               
    if isinstance(value, dict):                                                                                                           
        for key in ("mean_loss_ratio", "loss_ratio", "ratio"):                                                                            
            if _is_number(value.get(key)):                                                                                                
                return float(value[key])                                                                                                  
    return None                                                                                                                           
                                                                                                                                          
                                                                                                                                          
def condition_ratios(result):                                                                                                             
    """Find the per-condition ratios anywhere inside a contest_score result."""                                                           
    stack = [result]                                                                                                                      
    while stack:                                                                                                                          
        node = stack.pop()                                                                                                                
        if isinstance(node, dict):                                                                                                        
            if all(condition in node for condition in CONDITIONS):                                                                        
                ratios = {condition: _ratio_of(node[condition]) for condition in CONDITIONS}                                              
                if all(value is not None for value in ratios.values()):                                                                   
                    return ratios                                                                                                         
            stack.extend(node.values())                                                                                                   
        elif isinstance(node, list):                                                                                                      
            stack.extend(node)                                                                                                            
    return {condition: None for condition in CONDITIONS}                                                                                  
                                                                                                                                          
                                                                                                                                          
def technical_points(result):                                                                                                             
    for key in ("technical_points", "points", "technical_score", "points_total", "score"):                                                
        if _is_number(result.get(key)):                                                                                                   
            return float(result[key])                                                                                                     
    return None                                                                                                                           
                                                                                                                                          
                                                                                                                                          
def _fmt(value):                                                                                                                          
    return "n/a" if value is None else f"{value:.3f}"                                                                                     
                                                                                                                                          
                                                                                                                                          
def format_table(title, rows, names, max_points):                                                                                         
    header = ["candidate", f"points/{max_points}", "mean_ratio"] + list(CONDITIONS) + ["seconds"]                                         
    lines = [[*header]]                                                                                                                   
    for name in names:                                                                                                                    
        result = rows[name]["scores"][title]                                                                                              
        ratios = condition_ratios(result)                                                                                                 
        lines.append([                                                                                                                    
            name,                                                                                                                         
            _fmt(technical_points(result)),                                                                                               
            _fmt(result.get("mean_loss_ratio") if _is_number(result.get("mean_loss_ratio")) else None),                                   
            *[_fmt(ratios[c]) for c in CONDITIONS],                                                                                       
            f"{rows[name]['seconds']:.1f}",                                                                                               
        ])                                                                                                                                
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]                                                            
    out = [f"== {title.upper()} =="]                                                                                                      
    for index, line in enumerate(lines):                                                                                                  
        out.append("  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])                                                   
                             for i, cell in enumerate(line)))                                                                             
        if index == 0:                                                                                                                    
            out.append("  ".join("-" * w for w in widths))                                                                                
    return "\n".join(out)                                                                                                                 
                                                                                                                                          
                                                                                                                                          
def load_public(public):                                                                                                                  
    train_q, train_h = load_inputs(public / "train")                                                                                      
    val_q, val_h = load_inputs(public / "validation")                                                                                     
    train = Namespace(queries=train_q, history=train_h,                                                                                   
                      labels=read_labels(public / "train" / "labels.csv", train_q))                                                       
    validation = Namespace(queries=val_q, history=val_h,                                                                                  
                           targets=read_targets(public / "validation" / "targets.csv"))                                                   
    group_rows = read_rows(public / "validation" / "conditions.csv", ("query_id", "condition"))                                           
    groups = {row["query_id"]: row["condition"] for row in group_rows}                                                                    
    if len(groups) != len(group_rows):                                                                                                    
        raise ValidationError("Duplicate condition query ID.")                                                                            
    reference = read_submission(public / "validation" / "reference.csv")                                                                  
    ids = {q.query_id for q in val_q}                                                                                                     
    require_ids({t.query_id for t in validation.targets}, ids)                                                                            
    require_ids(groups, ids)                                                                                                              
    require_ids(reference, ids)                                                                                                           
    return train, validation, groups, reference, ids                                                                                      
                                                                                                                                          
                                                                                                                                          
def main(argv=None):                                                                                                                      
    available = registry()                                                                                                                
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)                           
    parser.add_argument("--data", type=Path, help="Path to a public/ folder.")                                                            
    parser.add_argument("--out", type=Path, help="New output folder (must not exist).")                                                   
    parser.add_argument("--candidates", nargs="*", default=None,                                                                          
                        help="Candidate names to run (default: all). Use --list to see names.")                                           
    parser.add_argument("--list", action="store_true", help="List candidate names and exit.")                                             
    parser.add_argument("--skip-audit", action="store_true",                                                                              
                        help="Score the tuning half only (hides audit and full-validation numbers).")                                     
    args = parser.parse_args(argv)                                                                                                        
                                                                                                                                          
    if args.list:                                                                                                                         
        print("\n".join(available))                                                                                                       
        return                                                                                                                            
    if args.data is None or args.out is None:                                                                                             
        parser.error("--data and --out are required.")                                                                                    
                                                                                                                                          
    names = list(available) if not args.candidates else list(args.candidates)                                                             
    unknown = [name for name in names if name not in available]                                                                           
    if unknown:                                                                                                                           
        parser.error(f"Unknown candidates: {', '.join(unknown)}. Available: {', '.join(available)}.")                                     
    if len(set(names)) != len(names):                                                                                                     
        parser.error("Candidate names must be unique.")                                                                                   
                                                                                                                                          
    public = args.data.resolve()                                                                                                          
    if args.out.exists():                                                                                                                 
        raise ValidationError("Choose a fresh --out directory; existing folders are never overwritten.")                                  
    metadata = round_metadata(public)                                                                                                     
    max_points = metadata["max_points"]                                                                                                   
                                                                                                                                          
    train, validation, groups, reference, ids = load_public(public)                                                                       
    tuning, audit = episode_folds(validation.targets, groups)                                                                             
    all_episodes = {t.episode_id for t in validation.targets}                                                                             
                                                                                                                                          
    args.out.mkdir(parents=True)                                                                                                          
    rows = {}                                                                                                                             
    for name in names:                                                                                                                    
        start = perf_counter()                                                                                                            
        model = available[name]().fit(train.queries, train.history, train.labels)                                                         
        prediction = model.predict(validation.queries, validation.history)                                                                
        require_ids(prediction, ids)                                                                                                      
        scores = {"tuning": subset_score(prediction, validation.targets, groups, reference, tuning, max_points)}                          
        if not args.skip_audit:                                                                                                           
            scores["audit"] = subset_score(prediction, validation.targets, groups, reference, audit, max_points)                          
            scores["full"] = subset_score(prediction, validation.targets, groups, reference, all_episodes, max_points)                    
        seconds = perf_counter() - start                                                                                                  
        rows[name] = {"scores": scores, "seconds": seconds}                                                                               
        (args.out / name).mkdir()                                                                                                         
        write_submission(args.out / name / "validation_predictions.csv", prediction)                                                      
        print(f"finished {name} in {seconds:.1f}s", file=sys.stderr)                                                                      
                                                                                                                                          
    sections = ["tuning"] if args.skip_audit else ["tuning", "audit", "full"]                                                             
    tables = [format_table(section, rows, names, max_points) for section in sections]                                                     
    text = "\n\n".join(tables)                                                                                                            
    print()                                                                                                                               
    print(text)                                                                                                                           
    if all(technical_points(rows[n]["scores"]["tuning"]) is None for n in names):                                                         
        print("\nNote: could not find technical points / condition ratios by name. "                                                      
              f"See {args.out / 'results.json'} for the raw contest_score output.", file=sys.stderr)                                      
                                                                                                                                          
    (args.out / "table.txt").write_text(text + "\n", encoding="utf-8")                                                                    
    write_json(args.out / "results.json", {                                                                                               
        "round": metadata["round"],                                                                                                       
        "season_id": metadata["season_id"],                                                                                               
        "max_points": max_points,                                                                                                         
        "candidates": names,                                                                                                              
        "tuning_episodes": sorted(tuning),                                                                                                
        "audit_episodes": None if args.skip_audit else sorted(audit),                                                                     
        "results": rows,                                                                                                                  
    })                                                                                                                                    
                                                                                                                                          
                                                                                                                                          
if __name__ == "__main__":                                                                                                                
    main()
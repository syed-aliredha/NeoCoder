import click
import os
import pandas as pd
from src.utils.configs import correctness_evaluation, technique_detection, calculate_creativity

@click.command()
@click.option("--task", type=click.Choice(["correctness", "detection", "creativity"]), help="Task to perform.")
@click.option("--inference-result-path", type=click.Path(exists=True), help="File Path of the inference result of dp dataset.")
@click.option("--human-solution-path", type=click.Path(exists=True), help="File Path of the human solutions", default=None)
@click.option("--test-case-path", type=click.Path(exists=True), help="File Path of the test case of dp dataset.", default=None)
@click.option("--save-folder", type=click.Path(), help="Folder to save the evaluation result.", default=None)

def main(
    task,
    inference_result_path,
    human_solution_path,
    test_case_path,
    save_folder
):
    if task == "detection":
        assert human_solution_path is not None, "Please provide human solution path."
        technique_detection(human_solution_path=human_solution_path,
                          inference_result_path=inference_result_path)
    
    elif task == "correctness":
        assert test_case_path is not None, "Please provide test case path."
        correctness_evaluation(inference_result_path=inference_result_path,
                             test_case_path=test_case_path,
                             save_folder=save_folder)
    
    elif task == "creativity":
        convergent_scores, divergent_scores, neogauge_scores = calculate_creativity(
            human_solution_path=human_solution_path,
            inference_result_path=inference_result_path,
            save_folder=save_folder
        )
        
        # Print detailed per-generation metrics
        print("\nPer Generation Creativity Metrics:")
        print("-" * 50)
        
        for gen_idx in range(len(convergent_scores)):
            print(f"\nGeneration {gen_idx}:")
            print(f"  Convergent Thinking: {convergent_scores[gen_idx]:.4f}")
            print(f"  Divergent Thinking:  {divergent_scores[gen_idx]:.4f}")
            print(f"  NeoGauge Score:     {neogauge_scores[gen_idx]:.4f}")
        
        # Print overall averages
        print("\nOverall Averages:")
        print("-" * 50)
        print(f"Average Convergent Thinking: {sum(convergent_scores)/len(convergent_scores):.4f}")
        print(f"Average Divergent Thinking:  {sum(divergent_scores)/len(divergent_scores):.4f}")
        print(f"Average NeoGauge Score:     {sum(neogauge_scores)/len(neogauge_scores):.4f}")
        
        # Print information about CSV files created
        model_name = os.path.basename(inference_result_path).split("_sample")[0]
        print("\nDetailed Results Saved:")
        print("-" * 50)
        print(f"1. Individual scores per problem and generation:")
        print(f"   {os.path.join(save_folder, f'{model_name}_individual_scores.csv')}")
        print(f"\n2. Aggregated metrics per generation:")
        print(f"   {os.path.join(save_folder, f'{model_name}_generation_metrics.csv')}")
        print(f"\n3. Pivot tables (problem × generation):")
        print(f"   Convergent: {os.path.join(save_folder, f'{model_name}_pivot_convergent.csv')}")
        print(f"   Divergent:  {os.path.join(save_folder, f'{model_name}_pivot_divergent.csv')}")
        print(f"   NeoGauge:   {os.path.join(save_folder, f'{model_name}_pivot_neogauge.csv')}")
        
        # Also save these metrics to a summary file
        if save_folder:
            summary_path = os.path.join(save_folder, f"{model_name}_metrics_summary.txt")
            
            with open(summary_path, 'w') as f:
                f.write("Per Generation Creativity Metrics:\n")
                f.write("-" * 50 + "\n")
                
                for gen_idx in range(len(convergent_scores)):
                    f.write(f"\nGeneration {gen_idx}:\n")
                    f.write(f"  Convergent Thinking: {convergent_scores[gen_idx]:.4f}\n")
                    f.write(f"  Divergent Thinking:  {divergent_scores[gen_idx]:.4f}\n")
                    f.write(f"  NeoGauge Score:     {neogauge_scores[gen_idx]:.4f}\n")
                
                f.write("\nOverall Averages:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Average Convergent Thinking: {sum(convergent_scores)/len(convergent_scores):.4f}\n")
                f.write(f"Average Divergent Thinking:  {sum(divergent_scores)/len(divergent_scores):.4f}\n")
                f.write(f"Average NeoGauge Score:     {sum(neogauge_scores)/len(neogauge_scores):.4f}\n")
                
                # Add information about the CSV files
                f.write("\nDetailed Results Saved:\n")
                f.write("-" * 50 + "\n")
                f.write(f"1. Individual scores per problem and generation:\n")
                f.write(f"   {os.path.join(save_folder, f'{model_name}_individual_scores.csv')}\n")
                f.write(f"\n2. Aggregated metrics per generation:\n")
                f.write(f"   {os.path.join(save_folder, f'{model_name}_generation_metrics.csv')}\n")
                f.write(f"\n3. Pivot tables (problem × generation):\n")
                f.write(f"   Convergent: {os.path.join(save_folder, f'{model_name}_pivot_convergent.csv')}\n")
                f.write(f"   Divergent:  {os.path.join(save_folder, f'{model_name}_pivot_divergent.csv')}\n")
                f.write(f"   NeoGauge:   {os.path.join(save_folder, f'{model_name}_pivot_neogauge.csv')}\n")
    
    else:
        raise ValueError("Invalid task.")

if __name__ == "__main__":
    main()
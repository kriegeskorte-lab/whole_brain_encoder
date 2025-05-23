#!/bin/bash

enc_output_layers=(1 3 5 7)
for subj in {1..1}; do
    for layer in "${enc_output_layers[@]}"; do
        for run in 1 2; do
    # for lr in "${lrs[@]}"; do

            # Define flags for each job
            subj_flag="--subj $subj"
            lr_flag="--lr 0.0001"
            batch_size_flag="--batch_size 16"
            enc_output_layer_flag="--enc_output_layer $layer"
            # parcel_dir_flag="--parcel_dir ./parcels/schaefer"
            run_flag="--run $run"
            backbone_flag="--backbone_arch radio"

            common_flags="$lr_flag $subj_flag $enc_output_layer_flag $run_flag $batch_size_flag $parcel_dir_flag $backbone_flag"
            echo "Common flags: $common_flags"

            flags_job1="$common_flags --hemi lh"
            flags_job2="$common_flags --hemi rh"

            # Submit the 4 independent jobs with their flags
            job1_id=$(sbatch --parsable train_passflags.sh $flags_job1)
            job2_id=$(sbatch --parsable train_passflags.sh $flags_job2)

            # Submit the dependent job
            final_job_id=$(sbatch --parsable --dependency=afterok:$job1_id:$job2_id plot_results.sh $subj_flag $enc_output_layer_flag $run_flag)

            # Output submitted jobs
            echo "Submitted jobs:"
            echo "Job 1 ID: $job1_id (Flags: $flags_job1)"
            echo "Job 2 ID: $job2_id (Flags: $flags_job2)"
            # echo "Job 3 ID: $job3_id (Flags: $flags_job3)"
            # echo "Job 4 ID: $job4_id (Flags: $flags_job4)"
            echo "Final job ID (dependent on all): $final_job_id"

        done
    done
done
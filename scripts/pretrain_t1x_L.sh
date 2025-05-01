python -m mepin.experiment.train --config-name=geodesic \
    project=$NEPTUNE_PROJECT \
    tags="[geodesic,t1x_xtb]" \
    dataset.data_dir="data/t1x_xtb" \
    dataset.swap_reactant_product=true \
    dataset.augment_angle_scale=0.05
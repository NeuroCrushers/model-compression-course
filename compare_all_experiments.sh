
for experiment in ./experiments/*
do
echo "Processing  ${experiment}..."
python inference.py --config_path ${experiment}/config.json --save_report True
done
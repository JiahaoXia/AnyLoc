cd /home/rise/XJH/Geo-Loc/AnyLoc

# Manville
echo "Starting feature extraction for Manville"
python feat_extraction_VLAD-DINOv2-wHydra.py --config-path /home/rise/XJH/Geo-Loc/config/DINOv2/satellite --config-name manville

# Hoboken
echo "Starting feature extraction for Hoboken"
python feat_extraction_VLAD-DINOv2-wHydra.py --config-path /home/rise/XJH/Geo-Loc/config/DINOv2/satellite --config-name hoboken

# Cranford
echo "Starting feature extraction for Cranford"
python feat_extraction_VLAD-DINOv2-wHydra.py --config-path /home/rise/XJH/Geo-Loc/config/DINOv2/satellite --config-name cranford
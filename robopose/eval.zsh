# DREAM datasets, ours (known joints)
python -m robopose.scripts.run_robot_eval --datasets=dream-panda --model=knownq --id 1804
python -m robopose.scripts.run_robot_eval --datasets=dream-baxter --model=knownq --id 1804
python -m robopose.scripts.run_robot_eval --datasets=dream-kuka --model=knownq --id 1804
# DREAM datasets, ours (unknown joints)
python -m robopose.scripts.run_robot_eval --datasets=dream-panda --model=unknownq --id 1804
python -m robopose.scripts.run_robot_eval --datasets=dream-baxter --model=unknownq --id 1804
python -m robopose.scripts.run_robot_eval --datasets=dream-kuka --model=unknownq --id 1804
import subprocess
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/Boosting/Tests/3_ensemble')
args = parser.parse_args()

csv_files = []
if len(csv_files) == 0:
    command = subprocess.Popen("find " + args.resultsDir + " -iname '*.csv'",
                               shell=True, stdout=subprocess.PIPE)
    for line in command.stdout.readlines():
        csv_files.append(line.decode("utf-8").replace('\n', ''))
print(csv_files)

results_folder = os.path.join(args.resultsDir, 'Results')
if os.path.isdir(results_folder):
    print('Directory exists. Removing directory tree {}'.format(results_folder))
    shutil.rmtree(results_folder)
print('Creating directory {}'.format(results_folder))
os.mkdir(results_folder)

for f in csv_files:
    d = f.split('_')[-1].split('.')[0]
    dd = os.path.join(results_folder, d)
    if not os.path.isdir(dd):
        print('Creating directory {}'.format(dd))
        os.mkdir(dd)
    ff = f.split('/')[-1].replace('_' + d + '.csv', '.txt')
    ff = os.path.join(dd, ff)
    shutil.copyfile(f, ff)

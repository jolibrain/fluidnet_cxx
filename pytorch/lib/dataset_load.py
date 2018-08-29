import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import glob
from .load_manta_data import loadMantaFile

class FluidNetDataset(Dataset):
    """Fluid Net dataset."""

    def __init__(self, conf, prefix, save_dt, preprocess=False, resume=False, pr_n_threads=0):

        self.conf = conf.copy()
        self.mconf = self.conf['modelParam']
        del self.conf['modelParam']

        self.prefix = prefix
        self.save_dt = save_dt
        self.data_dir = conf['dataDir']
        self.dataset = conf['dataset']
        self.n_threads = pr_n_threads # Num of threads to preprocess form .bin to .pt

        self.base_dir = self._get_base_dir(conf['dataDir'])
        self.scenes_folders = sorted(glob.os.listdir(self.base_dir))


        # Check number of scenes
        self.n_scenes = len(self.scenes_folders)
        self.scene_0 = int(self.scenes_folders[0])

        # Check how many timesteps per scene there are.
        self.step_per_scene = len(glob.glob(glob.os.path.join(self.base_dir, \
                                  self.scenes_folders[0], '*[0-9].bin')))

        self.pr_loader = loadMantaFile
        self.loader = torch.load

        # Log file (exists if dataset was preprocessed in the past)
        # It contains a dict with the name of data tensors and target tensors
        self.pr_log = {}
        self.logname = glob.os.path.join(self.data_dir, self.dataset, \
                'preprocessed_' + self.dataset + '_' + self.prefix + '.txt')

        # Pre-process
        if not resume:
            # Checking if pre-processing is needed:
            yes = {'yes','y', 'ye', ''}
            no = {'no','n'}
            if (glob.os.path.isfile(self.logname)):
                if preprocess:
                    print('For dataset ' + str(self.base_dir))
                    print('a log file exists showing a preprocessing in the past.')
                    self.pr_log = torch.load(self.logname)
                    print(self.pr_log)
                    print('Do you want to preprocess the dataset again? It takes some time. [y/n]')
                    choice = input().lower()
                    if choice in yes:
                        self.preprocess()
                    elif choice in no:
                        # We check if file exits in __getitem__
                        pass
                    else:
                        sys.stdout.write("Please respond with 'yes' or 'no'")
                else:
                    self.pr_log = torch.load(self.logname)
                    pass
            else:
                print('No log file found for ' + str(self.base_dir) \
                        + ' dataset. Preprocessing automatically.')
                self.preprocess()

        else:
            if (glob.os.path.isfile(self.logname)):
                print()
                print('For dataset ' + str(self.base_dir))
                print('a log file exists showing a preprocessing in the past.')
                print('OK to proceed with restarting')
                self.pr_log = torch.load(self.logname)
            else:
                print()
                print('No log file found, please create one by preprocessing the dataset. Set resume to false.')
                sys.exit()

        # Depending on inputs and loss, we will load different data in __getitem__
        self.nx = self.pr_log['nx']
        self.ny = self.pr_log['ny']
        self.nz = self.pr_log['nz']

        self.is3D = self.pr_log['is3D']
        self.inputChan = self.mconf['inputChannels']
        self.inDims = 1 # Flags is always passed

        if self.inputChan['div']:
            self.inDims += 1
        if self.inputChan['pDiv']:
            self.inDims += 1
        if self.inputChan['UDiv']:
            if self.is3D:
                self.inDims += 3
            else:
                self.inDims += 2

        self.mconf['inputDim'] = self.inDims
        self.mconf['is3D'] = self.is3D

    def createConfDict(self):
        return self.conf, self.mconf

    def preprocess(self):
        # Preprocess the dataset from .bin to .pt (much faster I/O)
        p = mp.Pool(self.n_threads)
        start = self.scene_0 * self.step_per_scene
        end = start + self.__len__()
        data_inputs = [idx for idx in range(start, end)]
        print('Pre-processing dataset:')
        try:
            p.map(self.__getitembin__, data_inputs)
            print('Pre-processing succeeded')
            self.pr_log = {'data': ['pDiv', 'UDiv', 'flagsDiv', 'densityDiv'],
                    'target' : ['p', 'U', 'density'], 'is3D' : False, 'nx' : 128,
                    'ny' : 128, 'nz' : 1}
            print('Log is now:')
            print(self.pr_log)
            torch.save(self.pr_log, self.logname)
        except:
            print('Pre-processing failed')

    def __len__(self):
        return self.n_scenes * self.step_per_scene

    # Used only in preprocessing
    def __getitembin__(self, idx):
        cur_scene = idx // self.step_per_scene
        cur_timestep = (idx % (self.step_per_scene)) * self.save_dt
        data_file = glob.os.path.join(self.base_dir, '{0:06d}'.format(cur_scene), \
                                      '{0:06d}.bin'.format(cur_timestep))
        data_div_file = glob.os.path.join(self.base_dir, '{0:06d}'.format(cur_scene), \
                                      '{0:06d}_divergent.bin'.format(cur_timestep))
        assert glob.os.path.isfile(data_file), 'Data file ' + data_file +  ' does not exists'
        assert glob.os.path.isfile(data_div_file), 'Data file does not exists'
        p, U, flags, density, is3D = self.pr_loader(data_file)
        pDiv, UDiv, flagsDiv, densityDiv, is3DDiv = self.pr_loader(data_div_file)

        assert is3D == is3DDiv, '3D flag is inconsistent!'
        assert torch.equal(flags, flagsDiv), 'Flags are not equal for idx ' + str(idx)

        data = torch.cat([pDiv, UDiv, flagsDiv, densityDiv, p, U, density], 1)


        save_file = glob.os.path.join(self.base_dir, '{0:06d}'.format(cur_scene), \
                                      '{0:06d}_pyTen.pt'.format(cur_timestep))
        torch.save(data, save_file)


    # Actual data loader
    def __getitem__(self, idx):
        cur_scene = idx // self.step_per_scene
        cur_timestep = (idx % (self.step_per_scene)) * self.save_dt
        data_file = glob.os.path.join(self.base_dir, '{0:06d}'.format(cur_scene), \
                                      '{0:06d}_pyTen.pt'.format(cur_timestep))
        assert glob.os.path.isfile(data_file), 'Data file ' + data_file +  ' does not exists'
        torch_file = self.loader(data_file)

        if (self.is3D):
            data = torch_file[0,0:6]
            target = torch_file[0,6:11]
        else:
            data = torch_file[0,0:5]
            target = torch_file[0,5:9]

      #  # data indexes     |           |
      #  #       (dim 1)    |    2D     |    3D
      #  # ----------------------------------------
      #  #   DATA:
      #  #       pDiv       |    0      |    0
      #  #       UDiv       |    1:3    |    1:4
      #  #       flags      |    3      |    4
      #  #       densityDiv |    4      |    5
      #  #   TARGET:
      #  #       p          |    5      |    6
      #  #       U          |    6:8    |    7:10
      #  #       density    |    8      |    10

        return data, target


    def _get_base_dir(self, data_dir):
        return glob.os.path.join(data_dir, self.dataset,  self.prefix)



from cProfile import label
import pickle
import os
import torchvision
import copy


class Logger(object):
    def __init__(self,
                 log_dir='./logs',
                 img_dir='./imgs',
                 monitoring=None,
                 monitoring_dir=None):
        self.stats = dict()
        self.log_dir = log_dir
        self.img_dir = img_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if not (monitoring is None or monitoring == 'none'):
            self.setup_monitoring(monitoring, monitoring_dir)
        else:
            self.monitoring = None
            self.monitoring_dir = None

    def setup_monitoring(self, monitoring, monitoring_dir=None):
        self.monitoring = monitoring
        self.monitoring_dir = monitoring_dir

        if monitoring == 'telemetry':
            import telemetry
            self.tm = telemetry.ApplicationTelemetry()
            if self.tm.get_status() == 0:
                print('Telemetry successfully connected.')
        elif monitoring == 'tensorboard':
            import tensorboardX
            self.tb = tensorboardX.SummaryWriter(monitoring_dir)
        else:
            raise NotImplementedError('Monitoring tool "%s" not supported!' %
                                      monitoring)

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

        k_name = '%s/%s' % (category, k)
        if self.monitoring == 'telemetry':
            self.tm.metric_push_async({'metric': k_name, 'value': v, 'it': it})
        elif self.monitoring == 'tensorboard':
            self.tb.add_scalar(k_name, v, it)

    def add_imgs(self, imgs, class_name, it):
        outdir = os.path.join(self.img_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%08d.png' % it)

        imgs = imgs / 2 + 0.5
        imgs = torchvision.utils.make_grid(imgs)
        torchvision.utils.save_image(copy.deepcopy(imgs), outfile, nrow=8)

        if self.monitoring == 'tensorboard':
            self.tb.add_image(class_name, copy.deepcopy(imgs), it)

    def add_label_hist(self, labels, class_name, it):
        import matplotlib.pyplot as plt
        import numpy as np

        outdir = os.path.join(self.img_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%08d.png' % it)

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        bottom_y = np.zeros(labels.shape[1])
        x = np.arange(labels.shape[1])
        width = 0.35
        for y in range(labels.shape[0]):
            ax.bar(x, labels[y], width, bottom=bottom_y, label=f"c {y:d}")
            bottom_y += labels[y]

        # ax.set_ylim(top=6000)
        ax.set_xticks(x)
        ax.tick_params(labelsize=16)
        
        fig.savefig(outfile, dpi=100)
        plt.clf(); plt.cla(); plt.close()

    def get_last(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        if not os.path.exists(filename):
            print('Warning: file "%s" does not exist!' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
        except EOFError:
            print('Warning: log file corrupted!')

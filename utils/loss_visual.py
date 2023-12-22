from torch.utils.tensorboard import SummaryWriter


class LossVisual:
      def __init__(self, store_path):
            self.path   = store_path
            self.writer = SummaryWriter(self.path)
            
      def add_scalar(self, loss, epoch):
            self.writer.add_scalar(self.path, loss.item(), epoch*200)

      def close(self):
            self.writer.close()
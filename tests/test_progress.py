from unittest import TestCase

import remotior_sensus
import time


class TestProgress(TestCase):

    def test_create(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test progress')

        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
        for e in range(0, 100):
            time.sleep(0.02)
            cfg.progress.update(
                message='test message',
                step=int(100 * (e + 1) / 100),
                percentage=int(100 * (e + 1) / 100)
            )
        cfg.progress.update(end=True)

        cfg.logger.log.debug('>>> test progress callback')
        rs.set(progress_callback=cfg.progress.print_progress)
        cfg.progress.update(
            process=__name__.split('.')[-1].replace('_', ' '),
            message='starting', start=True
        )
        for e in range(0, 100):
            time.sleep(0.02)
            cfg.progress.update(
                message='test message',
                step=int(100 * (e + 1) / 100),
                percentage=int(100 * (e + 1) / 100)
            )
        cfg.progress.update(end=True)

        # clear temporary directory
        rs.close()

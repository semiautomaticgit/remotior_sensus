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

        cfg.logger.log.debug('>>> test progress smtp')
        smtp_user = None
        smtp_password = None
        smtp_server = None
        smtp_recipients = None
        rs.set(smtp_user=smtp_user, smtp_password=smtp_password,
               smtp_server=smtp_server, smtp_recipients=smtp_recipients,
               smtp_notification=True)
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
        self.assertTrue(smtp_user is None)
        self.assertTrue(smtp_password is None)
        self.assertTrue(smtp_server is None)
        self.assertTrue(smtp_recipients is None)

        # clear temporary directory
        rs.close()

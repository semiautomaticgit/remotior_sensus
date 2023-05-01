from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import files_directories


class TestTableManager(TestCase):

    def test_table_manager(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        file1 = './data/files/file1.csv'
        file2 = './data/files/file2.csv'
        file3 = './data/files/file1.dbf'
        cfg.logger.log.debug('>>> test open file')
        matrix_file = rs.table_manager.open_file(
            file1, field_names=['id', 'main', 'field1', 'field2', 'field3',
                                'name']
        )
        self.assertGreater(len(matrix_file[0]), 0)
        cfg.logger.log.debug('>>> test open csv')
        matrix_file1 = rs.table_manager._open_csv(
            file1, field_name_list=['id', 'main', 'field1', 'field2', 'field3',
                                    'name']
        )
        self.assertGreater(len(matrix_file1[0]), 0)
        self.assertGreater(len(matrix_file1.id), 0)
        matrix_file2 = rs.table_manager._open_csv(file2)
        self.assertGreater(len(matrix_file2.value), 0)
        cfg.logger.log.debug('>>> test open dbf')
        matrix_file3 = rs.table_manager._open_dbf(
            file3, field_name_list=['id', 'main', 'field1', 'field2', 'field3',
                                    'name']
        )
        self.assertGreater(len(matrix_file3[0]), 0)
        self.assertGreater(len(matrix_file3.id), 0)
        cfg.logger.log.debug('>>> test join matrices')
        joined_table = rs.table_manager.join_matrices(
            matrix1=matrix_file1, matrix2=matrix_file2, field1_name='main',
            field2_name='value', join_type='leftouter', matrix1_postfix='_m1',
            matrix2_postfix='_m2'
        )
        self.assertGreater(len(joined_table.id_m1), 0)
        self.assertGreater(len(rs.table_manager.columns(joined_table)), 0)
        cfg.logger.log.debug('>>> test join tables')
        joined_table2 = rs.table_manager.join_tables(
            table1=matrix_file1, table2=matrix_file2, field1_name='main',
            field2_name='value', join_type='left'
        )
        self.assertGreater(len(joined_table2.id2), 0)
        self.assertGreater(len(rs.table_manager.columns(joined_table2)), 0)
        joined_test_outer = rs.table_manager.join_tables(
            table1=matrix_file1, table2=matrix_file2, field1_name='id',
            field2_name='id', join_type='outer'
        )
        self.assertGreater(len(joined_test_outer.id), 0)
        self.assertGreater(len(rs.table_manager.columns(joined_test_outer)), 0)
        joined_test_inner = rs.table_manager.join_tables(
            table1=matrix_file1, table2=matrix_file2, field1_name='main',
            field2_name='value', join_type='inner'
        )
        self.assertGreater(len(joined_test_inner.id2), 0)
        self.assertGreater(len(rs.table_manager.columns(joined_test_inner)), 0)
        joined_test_right = rs.table_manager.join_tables(
            table1=matrix_file1, table2=matrix_file2, field1_name='id',
            field2_name='id', join_type='right'
        )
        self.assertGreater(len(joined_test_right.id), 0)
        self.assertGreater(len(rs.table_manager.columns(joined_test_right)), 0)
        cfg.logger.log.debug('>>> test pivot_60')
        pivot1 = rs.table_manager.pivot_matrix(
            joined_table, row_field='value',
            column_function_list=[['field3_m1', 'sum']]
        )
        self.assertGreater(len(pivot1.field3_m1_sum), 0)
        cfg.logger.log.debug('>>> test rename field')
        renamed_field_pivot1 = rs.table_manager.rename_field(
            pivot1, 'field3_m1_sum', 'sum'
        )
        self.assertTrue(
            'sum' in rs.table_manager.columns(renamed_field_pivot1)
        )
        cfg.logger.log.debug('>>> test calculate')
        calculation = rs.table_manager.calculate(
            matrix=matrix_file3, expression_string='field.field3 * 1.5',
            output_field_name='calc'
        )
        self.assertGreater(len(calculation.calc), 0)
        calculation_multi = rs.table_manager.calculate_multi(
            matrix=matrix_file3,
            expression_string_list=['"field1" * 1.5', '"field2" * 3.5'],
            output_field_name_list=['calc1', 'calc2']
        )
        self.assertGreater(len(calculation_multi.calc1), 0)
        output = cfg.temp.temporary_file_path(name_suffix=cfg.csv_suffix)
        cfg.logger.log.debug('>>> matrix to csv')
        rs.table_manager.matrix_to_csv(
            matrix=calculation, output_path=output,
            fields=['name', 'id', 'calc'], nodata_value=4,
            nodata_value_output='nodata', separator=';', decimal_separator=','
        )
        self.assertTrue(files_directories.is_file(output))
        cfg.logger.log.debug('>>> test pivot_60')
        pivot2 = rs.table_manager.pivot_matrix(
            joined_table, row_field='name_m1',
            column_function_list=[['field3_m2', 'sum']]
        )
        cfg.logger.log.debug('>>> test get values')
        values = rs.table_manager.get_values(
            matrix=pivot2, value_field='field3_m2_sum',
            conditional_string='field.name_m1 == "a"'
        )
        self.assertGreater(len(values), 0)
        values2 = rs.table_manager.get_values(
            matrix=pivot2, value_field='name_m1',
            conditional_string='field.field3_m2_sum >100'
        )
        self.assertGreater(len(values2), 0)
        cfg.logger.log.debug('>>> test pivot_60')
        fields = rs.table_manager.pivot_matrix(
            joined_table, row_field='name_m1',
            secondary_row_field_list=['name_m2'],
            column_function_list=[['field3_m1', 'sum'], ['field3_m2', 'sum']],
            filter_string='matrix["field3_m1"] <= 1000', field_names=True
        )
        self.assertGreater(len(fields), 0)
        pivot3 = rs.table_manager.pivot_matrix(
            joined_table, row_field='name_m1',
            secondary_row_field_list=['name_m2'],
            column_function_list=[['field3_m1', 'sum'], ['field3_m2', 'sum']],
            filter_string='matrix["field3_m1"] <= 1000'
        )
        self.assertGreater(len(pivot3.name_m1), 0)
        cfg.logger.log.debug('>>> test redefine')
        redefined = rs.table_manager.redefine_matrix_columns(
            matrix=joined_table,
            input_field_names=['name_m1', 'field3_m1', 'name_m2'],
            output_field_names=['field1', 'field2', 'field3']
        )
        self.assertGreater(len(redefined.field1), 0)
        appended = rs.table_manager.append_values_to_table(
            matrix=redefined, value_list=[1, 2, 3]
        )
        self.assertGreater(len(appended.field1), 0)

        # clear temporary directory
        rs.close()

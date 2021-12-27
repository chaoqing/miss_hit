#!/usr/bin/env python3
##############################################################################
##                                                                          ##
##          MATLAB Independent, Small & Safe, High Integrity Tools          ##
##                                                                          ##
##              Copyright (C) 2020-2021, Florian Schanda                    ##
##                                                                          ##
##  This file is part of MISS_HIT.                                          ##
##                                                                          ##
##  MATLAB Independent, Small & Safe, High Integrity Tools (MISS_HIT) is    ##
##  free software: you can redistribute it and/or modify                    ##
##  it under the terms of the GNU Affero General Public License as          ##
##  published by the Free Software Foundation, either version 3 of the      ##
##  License, or (at your option) any later version.                         ##
##                                                                          ##
##  MISS_HIT is distributed in the hope that it will be useful,             ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of          ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           ##
##  GNU Afferto General Public License for more details.                    ##
##                                                                          ##
##  You should have received a copy of the GNU Affero General Public        ##
##  License along with MISS_HIT. If not, see                                ##
##  <http://www.gnu.org/licenses/>.                                         ##
##                                                                          ##
##############################################################################

from io import StringIO
from pathlib import Path
from collections import OrderedDict
import re

from miss_hit_core import pathutil
from miss_hit_core import command_line
from miss_hit_core import work_package
from miss_hit_core.m_ast import *
from miss_hit_core.errors import (Error,
                                  Message_Handler)
from miss_hit_core.m_lexer import MATLAB_Lexer
from miss_hit_core.m_parser import MATLAB_Parser


class Python_Visitor(AST_Visitor):
    """ Matlab To Python/Numpy output: Python """
    def __init__(self, fd, mh=None, matlab_alias='mp'):
        super().__init__()
        self.fd = fd
        self.mh = mh
        self.node_src = OrderedDict()
        pass_visitor = getattr(self, '_pass_visitor')
        self.node_visitor = {
            Function_Signature: pass_visitor,
            Action: pass_visitor,
        }

        self.func_alias = lambda n: f'{matlab_alias}.{n}'

    def __setitem__(self, node, src):
        self.node_src[node.uid] = src

    def pop(self, node):
        return self.node_src.pop(node.uid)

    @staticmethod
    def indent(src: str):
        return '\n'.join([f'    {l}' for l in src.split('\n')])

    def visit(self, node, n_parent, relation):
        pass

    def visit_end(self, node, n_parent, relation):
        visitor = self.node_visitor.get(node.__class__, None)
        try:
            if visitor is None:
                visitor = getattr(self, f'{node.__class__.__name__.lower()}_visitor')
                self.node_visitor[node.__class__] = visitor

            visitor(node, n_parent, relation)

        except AttributeError:
            self[node] = ''

        except NotImplementedError:
            self[node] = ''

            # Top of Node Root
        if n_parent is None:
            # TODO: remove redundant bracket with python formatter
            #     `black -i -S {}.py` can only solve tiny part of it
            src = self.pop(node) + "\n"
            if self.fd:
                self.fd.write(src)
            else:
                print(src)

    def _pass_visitor(self, node: Node, n_parent, relation):
        pass

    def general_for_statement_visitor(self, node: General_For_Statement, n_parent, relation):
        self[node] = f'for {self.pop(node.n_ident)} in {self.pop(node.n_expr)}:\n' \
            f'{self.indent(self.pop(node.n_body))}\n'

    def reshape_visitor(self, node: Reshape, n_parent, relation):
        self[node] = ':'

    def range_expression_visitor(self, node: Range_Expression, n_parent, relation):
        n_stride = "" if node.n_stride is None else f'{self.pop(node.n_stride)}, '
        self[node] = (f'colon('
                      f'{self.pop(node.n_first)}, '
                      f'{n_stride}'
                      f'{self.pop(node.n_last)})'
                      )

    def cell_reference_visitor(self, node: Cell_Reference, n_parent, relation):
        args = ', '.join(f'{self.pop(i)}-1' for i in node.l_args)
        self[node] = f'{self.pop(node.n_ident)}[{args}]'

    def __node_contains_end(self, node: (Identifier, Binary_Operation, Range_Expression)):
        if isinstance(node, Identifier):
            return node.t_ident.value == 'end'
        if isinstance(node, Range_Expression):
            return (self.__node_contains_end(node.n_first)
                    or self.__node_contains_end(node.n_last)
                    or self.__node_contains_end(node.n_stride))
        if isinstance(node, Binary_Operation):
            return (self.__node_contains_end(node.n_lhs)
                    or self.__node_contains_end(node.n_rhs))
        return False

    def reference_visitor(self, node: Reference, n_parent, relation):
        # TODO: determine reference is function call or slice
        is_index = False

        if any(isinstance(i, Reshape) for i in node.l_args):
            is_index = True
        elif any(self.__node_contains_end(i) for i in node.l_args):
            is_index = True
        elif n_parent is not None:
            l_lhs = []
            if isinstance(n_parent, Simple_Assignment_Statement):
                l_lhs.append(n_parent.n_lhs)
            elif isinstance(n_parent, Compound_Assignment_Statement):
                l_lhs.extend(n_parent.l_lhs)

            if any(i is node for i in l_lhs):
                is_index = True

        args = ', '.join(f'{self.pop(i)}{"-1" if is_index else ""}' for i in node.l_args)
        self[node] = f'{self.pop(node.n_ident)}{"[" if is_index else "("}{args}{"]" if is_index else ")"}'

    def identifier_visitor(self, node: Identifier, n_parent, relation):
        self[node] = node.t_ident.value

    def number_literal_visitor(self, node: Number_Literal, n_parent, relation):
        self[node] = node.t_value.value.replace('i', 'j')

    def char_array_literal_visitor(self, node: Char_Array_Literal, n_parent, relation):
        self[node] = f"'{node.t_string.value}'"

    def string_literal_visitor(self, node: String_Literal, n_parent, relation):
        self[node] = f"'{node.t_string.value}'"

    def special_block_visitor(self, node: Special_Block, n_parent, relation):
        raise NotImplementedError

    def entity_constraints_visitor(self, node: Entity_Constraints, n_parent, relation):
        raise NotImplementedError

    def function_file_visitor(self, node: Function_File, n_parent, relation):
        header = ('import mat2py.core as mp\n'
                  'from mat2py.core import (end, colon)\n')

        func = '\n'.join([self.pop(l) for l in node.l_functions])
        self[node] = '\n'.join([header, func])

    def script_file_visitor(self, node: Script_File, n_parent, relation):
        header = ('import mat2py.core as mp\n'
                  'from mat2py.core import (end, colon)\n')
        body = (f'def main():\n'
                f'{self.indent(self.pop(node.n_statements))}\n\n\n'
                f'if __name__ == "__main__":\n'
                f'{self.indent("main()")}')

        func = '\n'.join([self.pop(l) for l in node.l_functions])
        self[node] = '\n'.join([header, func, body])

    def sequence_of_statements_visitor(self, node: Sequence_Of_Statements, n_parent, relation):
        self[node] = '\n'.join([self.pop(l) for l in node.l_statements])

    def function_definition_visitor(self, node: Function_Definition, n_parent, relation):
        n_name = self.pop(node.n_sig.n_name)
        n_body = self.indent(self.pop(node.n_body))
        l_inputs = ', '.join([self.pop(i) for i in node.n_sig.l_inputs])
        l_outputs = self.indent('return {}'.format(', '.join([self.pop(i) for i in node.n_sig.l_outputs])))
        self[node] = f'def {n_name}({l_inputs}):\n{n_body}\n{l_outputs}\n'

    def compound_assignment_statement_visitor(self, node: Compound_Assignment_Statement, n_parent, relation):
        l_lhs = ', '.join([i if i!='~' else '_' for i in map(self.pop, node.l_lhs)])
        self[node] = f'{l_lhs} = {self.pop(node.n_rhs)}'

    def simple_assignment_statement_visitor(self, node: Simple_Assignment_Statement, n_parent, relation):
        self[node] = f'{self.pop(node.n_lhs)} = {self.pop(node.n_rhs)}'

    def function_call_visitor(self, node: Function_Call, n_parent, relation):
        args = ', '.join(self.pop(i) for i in node.l_args)
        self[node] = f'{self.pop(node.n_name)}({args})'

    def switch_statement_visitor(self, node: Switch_Statement, n_parent, relation):
        l_actions = []
        n_expr_l = self.pop(node.n_expr)

        bracket = ('', '') if isinstance(node.n_expr, (
            Identifier, Number_Literal,
            String_Literal, Char_Array_Literal,)) else ('(', ')')

        n_expr_l = f'{bracket[0]}{n_expr_l}{bracket[1]}'

        for i, a in enumerate(node.l_actions):
            key = "else" if a.n_expr is None else "elif" if i > 0 else "if"
            n_expr_r = '' if a.n_expr is None else self.pop(a.n_expr)
            bracket = ('', '') if isinstance(a.n_expr, (
                Identifier, Number_Literal,
                String_Literal, Char_Array_Literal,)) else ('(', ')')
            n_expr = '' if a.n_expr is None else f' {n_expr_l} == {bracket[0]}{n_expr_r}{bracket[1]}'
            n_body = self.pop(a.n_body)
            n_body = self.indent('pass' if len(n_body) == 0 else n_body)
            l_actions.append(f'{key}{n_expr}:\n{n_body}')

        self[node] = '\n'.join(l_actions)

    def if_statement_visitor(self, node: If_Statement, n_parent, relation):
        l_actions = []
        for i, a in enumerate(node.l_actions):
            key = "else" if a.n_expr is None else "elif" if i > 0 else "if"
            n_expr = '' if a.n_expr is None else ' '+self.pop(a.n_expr)
            n_body = self.pop(a.n_body)
            n_body = self.indent('pass' if len(n_body)==0 else n_body)
            l_actions.append(f'{key}{n_expr}:\n{n_body}')

        self[node] = '\n'.join(l_actions)

    def continue_statement_visitor(self, node: Continue_Statement, n_parent, relation):
        self[node] = 'continue'

    def break_statement_visitor(self, node: Break_Statement, n_parent, relation):
        self[node] = 'break'

    def return_statement_visitor(self, node: Return_Statement, n_parent, relation):
        self[node] = 'return'
        # TODO: fix the return value

    def row_visitor(self, node: Row, n_parent, relation):
        if len(node.l_items) == 1:
            self[node] = self.pop(node.l_items[0])
        else:
            self[node] = f'[{", ".join(self.pop(i) for i in node.l_items)}]'

    def row_list_visitor(self, node: Row_List, n_parent, relation):
        if len(node.l_items) == 0:
            self[node] = f'{self.func_alias("array")}([])'
        elif len(node.l_items) == 1:
            self[node] = self.pop(node.l_items[0])
        else:
            self[node] = f'{self.func_alias("stack")}(({", ".join(self.pop(i) for i in node.l_items)}))'

    def matrix_expression_visitor(self, node: Matrix_Expression, n_parent, relation):
        src = self.pop(node.n_content)
        self[node] = src if src.startswith(self.func_alias("")) else f'{self.func_alias("array")}({src})'
        # TODO: be careful with empty func_alias

    def unary_operation_visitor(self, node: Unary_Operation, n_parent, relation):
        t_op = node.t_op.value
        n_expr = self.pop(node.n_expr)

        bracket = ('', '') if isinstance(node.n_expr, (
            Identifier, Number_Literal,
            String_Literal, Char_Array_Literal,)) else ('(', ')')

        if t_op in ("'", ".'"):
            self[node] = f'{bracket[0]}{n_expr}{bracket[1]}.T'
        else:
            self[node] = f'{t_op}{bracket[0]}{n_expr}{bracket[1]}'

    def binary_logical_operation_visitor(self, node: Binary_Logical_Operation, n_parent, relation):
        self.binary_operation_visitor(node, n_parent, relation)

    def binary_operation_visitor(self, node: Binary_Operation, n_parent, relation):
        t_op = node.t_op.value
        n_lhs = self.pop(node.n_lhs)
        n_rhs = self.pop(node.n_rhs)

        if t_op in ('\\', '/', '*') and not isinstance(node.n_rhs, Number_Literal):
            func_name = {'\\': 'mldivide', '/': 'mrdivide', '*': 'dot'}[t_op]
            self[node] = f'{self.func_alias(func_name)}({n_lhs}, {n_rhs})'
            return

        t_op = {
            '~=': '!=', '&&': 'and', '||': 'or',
            './': '/', '\\': '/', '.*': '*', '.^': '^'
        }.get(t_op, t_op)

        bracket = [('', '') if isinstance(i, (
            Identifier, Number_Literal,
            String_Literal, Char_Array_Literal,
            Function_Call, Reference, Cell_Reference,
        ))
                   else ('(', ')') for i in (node.n_lhs, node.n_rhs)]

        self[node] = f'{bracket[0][0]}{n_lhs}{bracket[0][1]} {t_op} {bracket[1][0]}{n_rhs}{bracket[1][1]}'
        # TODO: replace operation to numpy format

    def import_statement_visitor(self, node: Import_Statement, n_parent, relation):
        raise NotImplementedError

    def metric_justification_pragma_visitor(self, node: Metric_Justification_Pragma, n_parent, relation):
        raise NotImplementedError

    def naked_expression_statement_visitor(self, node: Naked_Expression_Statement, n_parent, relation):
        self[node] = self.pop(node.n_expr)
        # TODO: determine a variable display or script call


class MH_Python_Result(work_package.Result):
    def __init__(self, wp, lines=None):
        super().__init__(wp, True)
        self.lines = lines


class MH_Python(command_line.MISS_HIT_Back_End):
    def __init__(self, options):
        super().__init__("MH Python")
        assert (isinstance(options.matlab_alias, str)
                and re.search(r'^[A-Za-z_][A-Za-z0-9_]*', options.matlab_alias))
        self.matlab_alias = options.matlab_alias
        self.python_alongside = options.python_alongside

    @classmethod
    def process_wp(cls, wp):
        # Create lexer
        lexer = MATLAB_Lexer(wp.mh,
                             wp.get_content(),
                             wp.filename,
                             wp.blockname)
        if wp.cfg.octave:
            lexer.set_octave_mode()
        if len(lexer.text.strip()) == 0:
            return MH_Python_Result(wp)

        # Create parse tree
        try:
            parser = MATLAB_Parser(wp.mh, lexer, wp.cfg)
            n_cu = parser.parse_file()
        except Error:
            return MH_Python_Result(wp)

        with StringIO() as fd:
            try:
                n_cu.visit(None, Python_Visitor(fd, wp.mh, matlab_alias=wp.options.matlab_alias), "Root")
                return MH_Python_Result(wp, fd.getvalue())
            except Error:
                return MH_Python_Result(wp)

    def process_result(self, result: MH_Python_Result):
        if not isinstance(result, MH_Python_Result):
            return
        if result.lines is None:
            return

        if self.python_alongside:
            with open(Path(result.wp.filename).with_suffix('.py'), 'w') as fp:
                fp.write(result.lines)
        else:
            print(result.lines)

    def post_process(self):
        pass


def main_handler():
    clp = command_line.create_basic_clp()

    # Extra language options
    clp["language_options"].add_argument(
        "--matlab-alias",
        default='mp',
        help="Matlab equivalent package name")

    # Extra output options
    clp["output_options"].add_argument(
        "--python-alongside",
        default=False,
        action="store_true",
        help="Create .py file alongside the .m file")

    # Extra debug options

    options = command_line.parse_args(clp)

    mh = Message_Handler("debug")

    mh.show_context = not options.brief
    mh.show_style   = False
    mh.show_checks  = True
    mh.autofix      = False

    python_backend = MH_Python(options)
    command_line.execute(mh, options, {}, python_backend)


def main():
    command_line.ice_handler(main_handler)


if __name__ == "__main__":
    main()

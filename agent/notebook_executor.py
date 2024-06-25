from typing import List
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
import io
import traceback
from autogen.coding import CodeBlock, CodeResult
from contextlib import redirect_stdout, redirect_stderr
from IPython import get_ipython

class NotebookExecutor:
    def __init__(self) -> None:
        self._ipython: InteractiveShell = get_ipython()
        # Store the original showtraceback method
        self._original_showtraceback = self._ipython.showtraceback

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        exitcode = 0

        # Custom output interceptor
        class OutputInterceptor:
            def __init__(self):
                self.content = ""
            def write(self, msg):
                self.content += msg
            def flush(self):
                pass

        # Custom showtraceback method that does nothing
        def custom_showtraceback(*args, **kwargs):
            pass

        for code_block in code_blocks:
            stdout_interceptor = OutputInterceptor()
            stderr_interceptor = OutputInterceptor()

            try:
                # Redirect stdout and stderr to our custom interceptors
                with redirect_stdout(stdout_interceptor), redirect_stderr(stderr_interceptor):
                    # Replace the showtraceback method
                    self._ipython.showtraceback = custom_showtraceback

                    result = self._ipython.run_cell(code_block.code, store_history=False)

                    # Restore the original showtraceback method
                    self._ipython.showtraceback = self._original_showtraceback

                # Process the intercepted output
                log += stdout_interceptor.content

                if result.error_in_exec:
                    error_type = type(result.error_in_exec).__name__
                    error_message = str(result.error_in_exec)
                    log += f"Error during execution: {error_type}: {error_message}\n"
                    exitcode = 1
                elif not result.success:
                    log += "Execution failed without specific error information.\n"
                    exitcode = 1
                elif result.result is not None:
                    log += str(result.result) + "\n"

            except Exception as e:
                log += f"Unexpected exception: {type(e).__name__}: {str(e)}\n"
                exitcode = 1
                break

        return CodeResult(exit_code=exitcode, output=log)
    
# class NotebookExecutor:
#     def __init__(self) -> None:
#         # Get the current IPython instance running in this notebook.
#         self._ipython: InteractiveShell = get_ipython()

#     def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
#         log = ""
#         exitcode = 0

#         for code_block in code_blocks:
#             stdout = io.StringIO()
#             stderr = io.StringIO()

#             try:
#                 # Redirect stdout and stderr to capture the output
#                 with redirect_stdout(stdout), redirect_stderr(stderr):
#                     result = self._ipython.run_cell(code_block.code, store_history=False)

#                 # Append the captured stdout and stderr to the log
#                 log += stdout.getvalue()
#                 log += stderr.getvalue()

#                 # Check if the execution resulted in an error
#                 if result.error_before_exec is not None:
#                     log += f"Error before execution: {result.error_before_exec}\n"
#                     exitcode = 1
#                 elif result.error_in_exec is not None:
#                     log += f"Error during execution: {result.error_in_exec}\n"
#                     exitcode = 1
#                 else:
#                     # Append the execution result (if any) to the log
#                     if result.result is not None:
#                         log += str(result.result)

#                 # If the result was not successful, set the exit code to 1
#                 if not result.success:
#                     exitcode = 1

#             except Exception as e:
#                 # Capture any exception raised during execution
#                 log += f"Exception occurred: {str(e)}\n"
#                 log += traceback.format_exc()
#                 exitcode = 1
#                 break

#         return CodeResult(exit_code=exitcode, output=log)

# class NotebookExecutor(CodeExecutor):

#     @property
#     def code_extractor(self) -> CodeExtractor:
#         # Extact code from markdown blocks.
#         return MarkdownCodeExtractor()

#     def __init__(self) -> None:
#         # Get the current IPython instance running in this notebook.
#         self._ipython = get_ipython()

#     def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
#         log = ""
#         for code_block in code_blocks:
#             result = self._ipython.run_cell("%%capture --no-display cap\n" + code_block.code,store_history=False)
#             # result = self._ipython.run_cell(code_block.code,store_history=False)
#             # result = self._ipython.run_cell(code_block.code,store_history=False)
#             log += self._ipython.ev("cap.stdout")
#             log += self._ipython.ev("cap.stderr")
#             # log += captured.stdout + captured.stderr
#             # if cap.stdout:
#             #     print(captured.stdout, end="")
#             # if captured.stderr:
#             #     log+=captured.stdout
#             if result.result is not None:
#                 log += str(result.result)
#             exitcode = 0 if result.success else 1
#             if result.error_before_exec is not None:
#                 log += f"Error before execution: {result.error_before_exec}\n"
#                 exitcode = 1
#             if result.error_in_exec is not None:
#                 log += f"Error during execution: {result.error_in_exec}\n"
#                 exitcode = 1
#             if exitcode != 0:
#                 break
#         return CodeResult(exit_code=exitcode, output=log)
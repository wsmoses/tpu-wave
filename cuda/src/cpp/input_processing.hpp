// Removed input processing declarations.
#ifndef INPUT_PROCESSING_H
#define INPUT_PROCESSING_H
#include <string>
// Dummy declarations to keep build system happy if needed
void file_input_processing ( std::string file_name );
void command_line_input_processing ( int argc, char* argv[], std::string argument_name );
void checking_and_printouts_input_parameters ();
#endif
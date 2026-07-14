/* bc_render.h — Diagnostic rendering for Booth.
 *
 * One place that turns a phase's bc_error_t[] into output a human actually
 * wants to read: the message, where it is, the offending line, and a caret
 * under the exact column. No malloc, fixed stack buffers, bounded. ANSI
 * colour only when stderr is a real terminal, so piped output stays clean. */

#ifndef BC_RENDER_H
#define BC_RENDER_H

#include "barracuda.h"   /* bc_error_t */

/* Render n diagnostics from one compilation phase to stderr.
 *   file : filename for the `-->` line (NULL becomes "<input>")
 *   src  : the buffer the error line numbers index into (NULL skips snippet)
 *   errs : the phase's error array
 *   n    : number of valid entries
 *
 * Each error prints as:
 *     error[E020]: expected ';', got '}'
 *       --> kernel.cu:12:14
 *        |
 *     12 |     int x = 5
 *        |              ^
 */
void bc_diag(const char *file, const char *src,
             const bc_error_t *errs, int n);

#endif /* BC_RENDER_H */

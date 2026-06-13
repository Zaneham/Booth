/* bc_render.c — Clang-style diagnostic rendering: message, location, the
 * offending source line, and a caret. Colour only on a real terminal. */

#define _POSIX_C_SOURCE 200809L   /* fileno, isatty: POSIX, not ISO C99 */

#include "bc_render.h"
#include <stdio.h>
#include <string.h>

#ifndef _WIN32
#include <unistd.h>   /* isatty, fileno */
#endif

/* Colour only on a tty, resolved once. Monochrome on Windows: ANSI there
 * needs a console-mode dance we skip. */
static int use_colour(void)
{
    static int resolved = 0;
    static int tty = 0;
    if (!resolved) {
        resolved = 1;
#ifndef _WIN32
        tty = isatty(fileno(stderr));
#endif
    }
    return tty;
}

/* Start and length of line `line` (1-based). By line number, not byte
 * offset, so it works whether or not a phase recorded one. */
static const char *line_at(const char *src, uint32_t line, int *len)
{
    const char *p   = src;
    uint32_t    cur = 1;
    while (cur < line && *p != '\0') {
        if (*p == '\n') cur++;
        p++;
    }
    const char *e = p;
    while (*e != '\0' && *e != '\n') e++;
    *len = (int)(e - p);
    return p;
}

static void render_one(const char *file, const char *src,
                       const bc_error_t *er)
{
    int c = use_colour();
    const char *red  = c ? "\033[1;31m" : "";
    const char *blue = c ? "\033[1;34m" : "";
    const char *rst  = c ? "\033[0m"    : "";

    fprintf(stderr, "%serror%s[E%03u]: %s\n",
            red, rst, (unsigned)er->eid, er->msg);
    fprintf(stderr, "  %s-->%s %s:%u:%u\n",
            blue, rst,
            file ? file : "<input>",
            er->loc.line, (unsigned)er->loc.col);

    if (!src) return;

    int         llen = 0;
    const char *lp   = line_at(src, er->loc.line, &llen);

    char num[16];
    int  nw = snprintf(num, sizeof num, "%u", er->loc.line);

    fprintf(stderr, "  %*s %s|%s\n", nw, "", blue, rst);
    fprintf(stderr, "  %s%s%s %s|%s %.*s\n",
            blue, num, rst, blue, rst, llen, lp);

    /* Pad to the column, tabs kept so the caret lands under tabbed source. */
    fprintf(stderr, "  %*s %s|%s ", nw, "", blue, rst);
    int pad = (er->loc.col > 0) ? (int)er->loc.col - 1 : 0;
    for (int i = 0; i < pad; i++)
        fputc((i < llen && lp[i] == '\t') ? '\t' : ' ', stderr);
    fprintf(stderr, "%s^%s\n", red, rst);
}

void bc_diag(const char *file, const char *src,
             const bc_error_t *errs, int n)
{
    if (!errs || n <= 0) return;
    for (int i = 0; i < n; i++)
        render_one(file, src, &errs[i]);
}

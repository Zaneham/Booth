/* Shared class declarations for the SYSPRINT demo. The kernel
 * uses these as compile-time constants; the host interns the
 * matching names in the same order so the IDs line up.
 *
 * If you add a class, add it in two places: here, and in the
 * BC_SP_INIT_DEMO_CLASSES macro at the bottom. There is no
 * compile-time check that the two are kept in sync. Pay
 * attention. */

#ifndef SYSPRINT_DEMO_CLASSES_H
#define SYSPRINT_DEMO_CLASSES_H

#define CLS_TRACE   1u
#define CLS_RESULT  2u
#define CLS_ERROR   3u

#define BC_SP_INIT_DEMO_CLASSES()                  \
    do {                                           \
        bc_sp_intern("DEMO.TRACE");                \
        bc_sp_intern("DEMO.RESULT");               \
        bc_sp_intern("DEMO.ERROR");                \
    } while (0)

#endif

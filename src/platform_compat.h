#ifndef PLATFORM_COMPAT_H
#define PLATFORM_COMPAT_H

#ifndef _WIN32

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdarg>

typedef int errno_t;

inline errno_t fopen_s(FILE** fp, const char* filename, const char* mode)
{
	if (fp == nullptr) {
		return EINVAL;
	}
	*fp = fopen(filename, mode);
	return (*fp == nullptr) ? errno : 0;
}

template <size_t N, typename... Args>
inline int sprintf_s(char (&buffer)[N], const char* format, Args... args)
{
	return snprintf(buffer, N, format, args...);
}

inline int sprintf_s(char* buffer, size_t size, const char* format, ...)
{
	va_list args;
	va_start(args, format);
	const int written = vsnprintf(buffer, size, format, args);
	va_end(args);
	return written;
}

template <size_t N>
inline errno_t strcat_s(char (&dest)[N], const char* src)
{
	const size_t len_dest = strlen(dest);
	const size_t len_src = strlen(src);
	if (len_dest + len_src + 1 > N) {
		return ERANGE;
	}
	strcat(dest, src);
	return 0;
}

#define sscanf_s sscanf

#endif

#endif

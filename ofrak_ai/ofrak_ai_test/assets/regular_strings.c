#include <stdio.h>
#include <stdlib.h>

int thisIsAReallyLongFunctionNameThatExceedsTheMinimumLength(int);

int main(void)
{
    char string[60] = "This is a test string that won't be sassified because it's not tagged as an AsciiString\n";
    printf("%s\n", string);
    int thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength = 1;
    printf("SXS: %s() ACTCTX_FLAG_RESOURCE_NAME_VALID set but lpResourceName == 0\n", "function1");
    if(thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength == 1)
    {
        thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;    
    }
    printf("SXS: %s() AssemblyDirectory is not null terminated\n", "function2");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("SXS: %s() NtCreateSection() failed. Status = 0x%x.\n", "function3", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("SXS: %s() Calling csrss server failed. Status = 0x%x\n", "function4", 0xdeadbeef);
    for(int j = 0; j < 10; j++)
    {
        thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1; 
    }
    printf("SXS: Invalid parameter(s) passed to FindActCtxSection\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("SXS: %s() CsrCaptureMessageMultiUnicodeStringsInPlace failed\n", "function5");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength = thisIsAReallyLongFunctionNameThatExceedsTheMinimumLength(thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength);
    printf("ConvertStringSecurityDescriptorToSecurityDescriptorW\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 2;
    printf("ConvertSecurityDescriptorToStringSecurityDescriptorW\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("SXS: %s - Failure getting active activation context\n", "function6");
    if(thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength < 0)
    {
        thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;    
    }
    printf("Kernel32: No mapping for ImageInformation.Machine == %04x\n", "dead");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("SXS: %s() NtQueryInformationFile failed. Status = 0x%x\n", "function7", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("GetSystemWindowsDirectory failed or the size was not adequate\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("Failed to get the paths for the crash vertical. Error was 0x%x\n", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("RtlWerpReportException failed with status code :%d. Will try to launch the process directly\n", 1337);
    switch(thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength)
    {
        case 0:
            thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength -= 1;
        case 1:
            thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength *= 2;
        default:
            thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    }
    printf("ReadProcessMemory failed while trying to read PebBaseAddress\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("WerpNtWow64ReadVirtualMemory64 failed while trying to read PebBaseAddress\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 2;
    printf("NtQueryInformationProcess failed with status: 0x%x\n", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("Stringcchcopy failed while copying the debugger path 0x%x\n", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("StringCchPrintf failed while printing the debugger path with 0x%x\n", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("StringCchPrintf failed while printng the debugger commandline with 0x%x\n", 0xdeadbeef);
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("Exception encountered while running the recovery function\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength += 1;
    printf("No recovery rotuine found when control reached WerpRecoveryInvokedRemotely\n");
    thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength *= 2;
    return thisIsAReallyLongIdentifierNameThatExceedsTheMinimumLength;
}

int thisIsAReallyLongFunctionNameThatExceedsTheMinimumLength(int i)
{
    i *= 2;
    return i;
}

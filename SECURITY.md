# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it
privately through GitHub instead: on this repository, go to the
**Security** tab and choose **Report a vulnerability**.

Include what an attacker controls (for example a scan file someone opens),
what they can achieve, and the steps to reproduce it. A small file that shows
the problem helps most.

You will get an answer within a week. Splat Repair is maintained by one
person, so a fix can take longer; you will hear how it is going.

## What counts

Splat Repair opens scan files that people download from anywhere, so
anything a crafted `.ply`, `.splat` or `.spz` file can do beyond showing
wrong pixels is in scope: running code, reading or writing files the user did
not choose, or hanging or crashing the machine.

## Supported versions

Only the latest release gets security fixes.

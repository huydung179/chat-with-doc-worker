export const HTTP_STATUS_RESPONSES = Object.freeze({
    OK: { status: 200, message: "Success", action: "No additional action is required.", code: "OK" },
    NOT_FOUND: { status: 404, message: "Not found", action: "Check the URL or requested resource.", code: "NOT_FOUND" },
    UNAUTHORIZED: { status: 401, message: "Unauthorized", action: "Please login to continue.", code: "UNAUTHORIZED" },
    FORBIDDEN: { status: 403, message: "Forbidden", action: "Contact the administrator to get access permission.", code: "FORBIDDEN" },
    BAD_REQUEST: { status: 400, message: "Bad request", action: "Check the input information and try again.", code: "BAD_REQUEST" },
    INTERNAL_SERVER_ERROR: { status: 500, message: "Internal server error", action: "Please try again later or contact the technical support.", code: "INTERNAL_SERVER_ERROR" },
    CONFLICT: { status: 409, message: "Data already exists", action: "Please check and use different information.", code: "CONFLICT" },
})
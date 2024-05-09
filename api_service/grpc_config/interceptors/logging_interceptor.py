import logging
import grpc

class LoggingInterceptor(grpc.ServerInterceptor):  # pylint: disable=too-few-public-methods
    """
    A gRPC server interceptor that logs the details of each incoming request.

    This interceptor logs the method name of each incoming request.
    """
    def intercept_service(self, continuation, handler_call_details):
        # Log the incoming request
        logging.info('Received request for method %s', handler_call_details.method)

        # Continue with the request
        handler = continuation(handler_call_details)

        return handler

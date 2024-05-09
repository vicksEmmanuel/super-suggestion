import grpc
from grpc import StatusCode


class AuthInterceptor(grpc.ServerInterceptor):
    def __init__(self, exclude_routes=None):
        self.exclude_routes = exclude_routes if exclude_routes else []

    def intercept_service(self, continuation, handler_call_details):
        # Check if the current route is in the exclude list
        if handler_call_details.method not in self.exclude_routes:
            metadata = dict(handler_call_details.invocation_metadata)
            if 'authorization' not in metadata:
                def deny(_, context):
                    context.abort(StatusCode.UNAUTHENTICATED, 'Auth is invalid or missing')
                return grpc.unary_unary_rpc_method_handler(deny)
            else:
                # TODO: Check the database to see if they have a valid API key
                pass
        return continuation(handler_call_details)

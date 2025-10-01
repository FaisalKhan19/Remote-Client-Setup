import hashlib
import hmac
import time
import requests
from typing import Optional, Dict, Any
from urllib.parse import urlencode


class AsterDEXClient:
    """
    Aster DEX Futures API client for opening limit orders and closing positions.
    
    Based on Aster DEX API documentation: https://docs.asterdex.com/product/aster-pro/api/api-documentation
    """
    
    def __init__(self, api_key: str, api_secret: str, proxy_url:str=None, base_url: str = "https://fapi.asterdex.com"):
        """
        Initialize the Aster DEX client.
        
        Args:
            api_key: Your API key from Aster DEX
            api_secret: Your API secret from Aster DEX
            base_url: Base URL for API requests (default: https://fapi.asterdex.com)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

        if proxy_url:
            self.has_proxy = True
        else:
            self.has_proxy = False

        self.proxies = None

        self.time_offset = 0  # Time offset from server
        self._sync_time()  # Sync time on initialization

        self.n_req = 0

        
    def _create_signature(self, params: str) -> str:
        """Create HMAC SHA256 signature for authenticated requests."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _sync_time(self):
        """Sync local time with server time."""
        try:
            response = requests.get(f"{self.base_url}/fapi/v1/time", proxies=self.proxies)
            server_time = response.json()['serverTime']
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            print(f"Time synced. Offset: {self.time_offset}ms")
        except Exception as e:
            print(f"Warning: Could not sync time with server: {e}")
            self.time_offset = 0

    def _get_timestamp(self) -> int:
        """Get current timestamp adjusted for server time."""
        return int(time.time() * 1000) + self.time_offset

    def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = False, proxied: bool = False) -> Dict[str, Any]:
        """
        Make HTTP request to Aster DEX API.
        """
        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}
        
        if params is None:
            params = {}
            
        if signed:
            # Use synced timestamp
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = 10000  # 10 seconds
            
            # Sign the payload
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, proxies=self.proxies)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers, proxies=self.proxies)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, proxies=self.proxies)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            self.n_req += 1
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_msg += f"\nResponse: {e.response.text}"
            raise Exception(error_msg)
    
    def _sign_payload(self, params: dict) -> dict:
        """
        Signs the given params with HMAC SHA256 using SECRET_KEY
        """
        # step 1: build query string in exact order you send
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        
        # step 2: create signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # step 3: add signature
        params["signature"] = signature
        return params

    def _place_order(self, symbol="BTCUSDT", side="BUY", type="LIMIT", qty=0.001, price=100000):
        url = f"{self.base_url}/fapi/v1/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": type,
            "timeInForce": "GTC",
            "quantity": qty,
            "price": price,
            "recvWindow": 8000,
            "timestamp": int(time.time() * 1000)  # must be ms
        }
        signed_params = self._sign_payload(params)
        if type == 'MARKET':
            params.pop('timeInForce')
            params.pop('price')
        headers = {"X-MBX-APIKEY": self.api_key}
        r = requests.post(url, headers=headers, params=signed_params, proxies=self.proxies)
        return r.json()
    
    def place_market_order(self, symbol: str, side: str, quantity: float, 
                        position_side: Optional[str] = None,
                        reduce_only: bool = False,
                        new_order_resp_type: str = "ACK", proxied: bool = False) -> Dict[str, Any]:
        """
        Place a market order on Aster DEX.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            side: Order side - "BUY" or "SELL"
            quantity: Order quantity
            position_side: Position side - "BOTH" (One-way Mode) or "LONG"/"SHORT" (Hedge Mode)
                        Default is "BOTH" if not specified
            reduce_only: Whether this is a reduce-only order (default: False)
                        Cannot be used in Hedge Mode
            new_order_resp_type: Response type - "ACK" or "RESULT" (default: "ACK")
                            "RESULT" returns the final filled result
            
        Returns:
            API response with order details
            
        Example:
            # Buy market order
            client.place_market_order("BTCUSDT", "BUY", 0.001)
            
            # Sell market order with position side (Hedge Mode)
            client.place_market_order("BTCUSDT", "SELL", 0.001, position_side="LONG")
            
            # Reduce-only sell order
            client.place_market_order("BTCUSDT", "SELL", 0.001, reduce_only=True)
        """
        url = f"{self.base_url}/fapi/v1/order"
        
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "timestamp": self._get_timestamp(),
            "recvWindow": 7000
        }
        
        # Add optional parameters
        if position_side:
            params["positionSide"] = position_side.upper()
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        if new_order_resp_type:
            params["newOrderRespType"] = new_order_resp_type.upper()
        
        # Sign the payload
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        
        headers = {"X-MBX-APIKEY": self.api_key}
        response = requests.post(url, headers=headers, params=params, proxies=self.proxies)
        self.n_req += 1
        try:
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            error_msg = f"Market order failed: {e}"
            if hasattr(response, 'text'):
                error_msg += f"\nResponse: {response.text}"
            raise Exception(error_msg)
    
    def open_limit_long_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        """
        Open a long position with a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            quantity: Order quantity
            price: Limit price            
        Returns:
            Order response from API
        """
            
        return self._place_order(symbol, "BUY", 'LIMIT', quantity, price)
    
    def open_limit_short_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        """
        Open a short position with a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Order response from API
        """
            
        return self._place_order(symbol, "SELL", 'LIMIT', quantity, price)
    
    def close_position_market(
        self,
        symbol: str,
        position_side: str = "BOTH",
        quantity: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Close position using market order.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side to close (LONG, SHORT, or BOTH)
            quantity: Specific quantity to close (optional, closes all if not specified)
            
        Returns:
            Order response from API
        """
        # Get current position to determine side
        positions = self.get_position_info(symbol)
        
        if not positions:
            raise Exception(f"No position found for symbol {symbol}")
            
        position = positions[0] if len(positions) == 1 else next(
            (p for p in positions if p['positionSide'] == position_side), None
        )
        
        if not position:
            raise Exception(f"No {position_side} position found for {symbol}")
            
        position_amt = float(position['positionAmt'])
        
        if position_amt == 0:
            raise Exception(f"Position amount is zero for {symbol}")
        
        # Determine order side (opposite of position)
        if position_amt > 0:  # Long position
            side = 'SELL'
        else:  # Short position
            side = 'BUY'
            position_amt = abs(position_amt)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': str(quantity) if quantity else str(position_amt),
            'positionSide': position_side,
            'reduceOnly': 'true'
        }
        
        return self._make_request('POST', '/fapi/v1/order', params, signed=True, **kwargs)
    
    def close_position_limit(
        self,
        symbol: str,
        price: float,
        position_side: str = "BOTH",
        quantity: Optional[float] = None,
        time_in_force: str = "GTC", **kwargs
    ) -> Dict[str, Any]:
        """
        Close position using limit order.
        
        Args:
            symbol: Trading pair symbol
            price: Limit price for closing
            position_side: Position side to close (LONG, SHORT, or BOTH)
            quantity: Specific quantity to close (optional, closes all if not specified)
            time_in_force: Time in force (GTC, IOC, FOK, GTX)
            
        Returns:
            Order response from API
        """
        # Get current position to determine side
        positions = self.get_position_info(symbol)
        
        if not positions:
            raise Exception(f"No position found for symbol {symbol}")
            
        position = positions[0] if len(positions) == 1 else next(
            (p for p in positions if p['positionSide'] == position_side), None
        )
        
        if not position:
            raise Exception(f"No {position_side} position found for {symbol}")
            
        position_amt = float(position['positionAmt'])
        
        if position_amt == 0:
            raise Exception(f"Position amount is zero for {symbol}")
        
        # Determine order side (opposite of position)
        if position_amt > 0:  # Long position
            side = 'SELL'
        else:  # Short position
            side = 'BUY'
            position_amt = abs(position_amt)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'quantity': str(quantity) if quantity else str(position_amt),
            'price': str(price),
            'timeInForce': time_in_force,
            'positionSide': position_side,
            'reduceOnly': 'true'
        }
        
        return self._make_request('POST', '/fapi/v1/order', params, signed=True, **kwargs)
    
    def close_all_positions_market(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Close all positions for a symbol using market order with closePosition=true.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Order response from API
        """
        # Get current positions to determine if we have long or short
        positions = self.get_position_info(symbol)
        
        if not positions:
            raise Exception(f"No positions found for symbol {symbol}")
        
        # Check if we have any non-zero positions
        active_positions = [p for p in positions if float(p['positionAmt']) != 0]
        
        if not active_positions:
            raise Exception(f"No active positions found for {symbol}")
        
        # For close all, we need to determine the dominant side
        total_amt = sum(float(p['positionAmt']) for p in active_positions)
        
        if total_amt > 0:
            side = 'SELL'  # Close long positions
        elif total_amt < 0:
            side = 'BUY'   # Close short positions
        else:
            raise Exception("Net position is zero")
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'closePosition': 'true'
        }
        
        return self._make_request('POST', '/fapi/v1/order', params, signed=True, **kwargs)
    
    def get_position_info(self, symbol: Optional[str] = None, **kwargs) -> list:
        """
        Get position information.
        
        Args:
            symbol: Trading pair symbol (optional, returns all if not specified)
            
        Returns:
            List of position information
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return self._make_request('GET', '/fapi/v2/positionRisk', params, signed=True, **kwargs)
    
    def cancel_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Cancel an active order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID (either order_id or client_order_id required)
            client_order_id: Client order ID
            
        Returns:
            Cancellation response from API
        """
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = str(order_id)
        if client_order_id:
            params['origClientOrderId'] = client_order_id
            
        return self._make_request('DELETE', '/fapi/v1/order', params, signed=True, **kwargs)
    
    def get_open_orders(self, symbol: str, **kwargs) -> list:
        """
        Get all open orders.
        
        Args:
            symbol: Trading pair symbol (optional, returns all if not specified)
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return self._make_request('GET', '/fapi/v1/openOrders', params, signed=True, **kwargs)
    
    def set_leverage(self, multiplier:int, symbol: str, **kwargs):
        """
        Change the initial leverage on a symbol

        Args:
            multiplier: Leverage amount (eg. x10, x20, x50.....)
            symbol: Trading pair symbol to change leverage for

        Returns:
            Response from API
        """
        params = {
            'symbol': symbol,
            'leverage': multiplier
        }

        return self._make_request('POST', '/fapi/v1/leverage', params, signed=True, **kwargs)
    
    def change_leverage(self, symbol: str, leverage: int, proxied:bool=False) -> Dict[str, Any]:
        """
        Change the initial leverage for a specific symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            leverage: Target leverage (1 to 125)
            
        Returns:
            API response with leverage details
            
        Example:
            # Set 10x leverage for BTCUSDT
            result = client.change_leverage("BTCUSDT", 10)
            print(f"Leverage set to {result['leverage']}x")
            
            # Set 50x leverage
            result = client.change_leverage("BTCUSDT", 50)
        """
        if not 1 <= leverage <= 125:
            raise ValueError("Leverage must be between 1 and 125")
        
        url = f"{self.base_url}/fapi/v1/leverage"
        
        params = {
            'symbol': symbol,
            'leverage': leverage,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 9000
        }
        
        # Sign the payload
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        
        headers = {"X-MBX-APIKEY": self.api_key}
        response = requests.post(url, headers=headers, params=params, proxies=self.proxies)
        
        try:
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            error_msg = f"Change leverage failed: {e}"
            if hasattr(response, 'text'):
                error_msg += f"\nResponse: {response.text}"
            raise Exception(error_msg)
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances and positions.
        
        Returns:
            Account information
        """
        return self._make_request('GET', '/fapi/v4/account', signed=True)
    
    def get_symbol_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            
        Returns:
            Current price as float
        """
        params = {'symbol': symbol}
        response = self._make_request('GET', '/fapi/v1/ticker/price', params)
        return float(response['price'])
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Get current prices for all symbols.
        
        Returns:
            Dictionary mapping symbol to price
        """
        response = self._make_request('GET', '/fapi/v1/ticker/price')
        return {item['symbol']: float(item['price']) for item in response}
    
    def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker statistics for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker information including price, volume, change, etc.
        """
        params = {'symbol': symbol}
        response = self._make_request('GET', '/fapi/v1/ticker/24hr', params)
        
        # Convert numeric fields to appropriate types
        numeric_fields = [
            'priceChange', 'priceChangePercent', 'weightedAvgPrice', 
            'prevClosePrice', 'lastPrice', 'lastQty', 'openPrice', 
            'highPrice', 'lowPrice', 'volume', 'quoteVolume'
        ]
        
        for field in numeric_fields:
            if field in response:
                response[field] = float(response[field])
                
        return response
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of entries to return (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            Order book with bids and asks
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }
        response = self._make_request('GET', '/fapi/v1/depth', params)
        
        # Convert price and quantity strings to floats
        response['bids'] = [[float(price), float(qty)] for price, qty in response['bids']]
        response['asks'] = [[float(price), float(qty)] for price, qty in response['asks']]
        
        return response
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Args:
            symbol: Specific symbol to get info for (optional)
            
        Returns:
            Exchange information including filters and trading rules
        """
        response = self._make_request('GET', '/fapi/v1/exchangeInfo')
        
        if symbol:
            # Filter for specific symbol
            symbols = response.get('symbols', [])
            symbol_info = next((s for s in symbols if s['symbol'] == symbol), None)
            if not symbol_info:
                raise Exception(f"Symbol {symbol} not found")
            return symbol_info
            
        return response
    
    def get_symbol_filters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Get trading filters for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary of filters with filter type as key
        """
        symbol_info = self.get_exchange_info(symbol)
        filters = {}
        
        for filter_info in symbol_info.get('filters', []):
            filter_type = filter_info['filterType']
            filters[filter_type] = filter_info
            
        return filters
    
    def get_min_trade_quantity(self, symbol: str) -> float:
        """
        Get minimum trade quantity for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Minimum trade quantity as float
        """
        filters = self.get_symbol_filters(symbol)
        
        lot_size_filter = filters.get('LOT_SIZE')
        if lot_size_filter:
            return float(lot_size_filter['minQty'])
            
        raise Exception(f"LOT_SIZE filter not found for symbol {symbol}")
    
    def get_min_trade_value(self, symbol: str) -> float:
        """
        Get minimum notional (trade value) for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Minimum notional value as float
        """
        filters = self.get_symbol_filters(symbol)
        
        min_notional_filter = filters.get('MIN_NOTIONAL')
        if min_notional_filter:
            return float(min_notional_filter['notional'])
            
        raise Exception(f"MIN_NOTIONAL filter not found for symbol {symbol}")
    
    def get_price_precision(self, symbol: str) -> int:
        """
        Get price precision (decimal places) for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of decimal places for price
        """
        symbol_info = self.get_exchange_info(symbol)
        return symbol_info.get('pricePrecision', 8)
    
    def get_quantity_precision(self, symbol: str) -> int:
        """
        Get quantity precision (decimal places) for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of decimal places for quantity
        """
        symbol_info = self.get_exchange_info(symbol)
        return symbol_info.get('quantityPrecision', 8)
    
    def get_step_size(self, symbol: str) -> float:
        """
        Get step size for quantity increments.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Step size for quantity
        """
        filters = self.get_symbol_filters(symbol)
        
        lot_size_filter = filters.get('LOT_SIZE')
        if lot_size_filter:
            return float(lot_size_filter['stepSize'])
            
        raise Exception(f"LOT_SIZE filter not found for symbol {symbol}")
    
    def get_tick_size(self, symbol: str) -> float:
        """
        Get tick size for price increments.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tick size for price
        """
        filters = self.get_symbol_filters(symbol)
        
        price_filter = filters.get('PRICE_FILTER')
        if price_filter:
            return float(price_filter['tickSize'])
            
        raise Exception(f"PRICE_FILTER not found for symbol {symbol}")
    
    def get_symbol_trading_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive trading information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with all trading parameters
        """
        try:
            current_price = self.get_symbol_price(symbol)
            min_quantity = self.get_min_trade_quantity(symbol)
            min_notional = self.get_min_trade_value(symbol)
            step_size = self.get_step_size(symbol)
            tick_size = self.get_tick_size(symbol)
            price_precision = self.get_price_precision(symbol)
            quantity_precision = self.get_quantity_precision(symbol)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'min_quantity': min_quantity,
                'min_notional': min_notional,
                'step_size': step_size,
                'tick_size': tick_size,
                'price_precision': price_precision,
                'quantity_precision': quantity_precision,
                'min_trade_value_usdt': min_notional,
                'min_quantity_for_current_price': max(min_quantity, min_notional / current_price)
            }
            
        except Exception as e:
            raise Exception(f"Failed to get trading info for {symbol}: {e}")
    
    def calculate_valid_quantity(self, symbol: str, desired_quantity: float) -> float:
        """
        Calculate valid quantity based on symbol's step size.
        
        Args:
            symbol: Trading pair symbol
            desired_quantity: Desired quantity to trade
            
        Returns:
            Valid quantity that meets exchange requirements
        """
        min_qty = self.get_min_trade_quantity(symbol)
        step_size = self.get_step_size(symbol)
        
        if desired_quantity < min_qty:
            return min_qty
            
        # Round down to nearest step size
        steps = int((desired_quantity - min_qty) / step_size)
        valid_quantity = min_qty + (steps * step_size)
        
        # Round to appropriate precision
        precision = self.get_quantity_precision(symbol)
        return round(valid_quantity, precision)
    
    def calculate_valid_price(self, symbol: str, desired_price: float) -> float:
        """
        Calculate valid price based on symbol's tick size.
        
        Args:
            symbol: Trading pair symbol
            desired_price: Desired price
            
        Returns:
            Valid price that meets exchange requirements
        """
        filters = self.get_symbol_filters(symbol)
        price_filter = filters.get('PRICE_FILTER')
        
        if not price_filter:
            raise Exception(f"PRICE_FILTER not found for symbol {symbol}")
            
        min_price = float(price_filter['minPrice'])
        max_price = float(price_filter['maxPrice'])
        tick_size = float(price_filter['tickSize'])
        
        # Check bounds
        if desired_price < min_price:
            return min_price
        if max_price > 0 and desired_price > max_price:
            return max_price
            
        # Round to nearest tick size
        if tick_size > 0:
            ticks = round((desired_price - min_price) / tick_size)
            valid_price = min_price + (ticks * tick_size)
        else:
            valid_price = desired_price
            
        # Round to appropriate precision
        precision = self.get_price_precision(symbol)
        return round(valid_price, precision)


# Example usage
def example_usage():
    """
    Example usage of the Aster DEX API functions.
    Replace with your actual API credentials.
    """
    # Initialize client
    client = AsterDEXClient(
        api_key="place_api_key_here",
        api_secret="place_api_secret_here"
    )
    
    try:
        # Example 1: Get current price
        print("Getting current price...")
        btc_price = client.get_symbol_price("BTCUSDT")
        print(f"BTC current price: ${btc_price:,.2f}")
        
        # Example 2: Get comprehensive trading info
        print("\nGetting trading info...")
        trading_info = client.get_symbol_trading_info("BTCUSDT")
        print(f"Trading info for BTCUSDT:")
        for key, value in trading_info.items():
            print(f"  {key}: {value}")
        
        # Example 3: Get minimum trade quantity
        print(f"\nMinimum trade quantity: {client.get_min_trade_quantity('BTCUSDT')}")
        print(f"Minimum trade value: ${client.get_min_trade_value('BTCUSDT')}")
        
        # Example 4: Calculate valid order parameters
        print("\nCalculating valid order parameters...")
        desired_qty = 0.00123456
        valid_qty = client.calculate_valid_quantity("BTCUSDT", desired_qty)
        print(f"Desired quantity: {desired_qty}, Valid quantity: {valid_qty}")
        
        desired_price = 49999.123456
        valid_price = client.calculate_valid_price("BTCUSDT", desired_price)
        print(f"Desired price: {desired_price}, Valid price: {valid_price}")
        
        # Example 5: Get 24hr ticker
        print("\nGetting 24hr ticker...")
        ticker = client.get_symbol_ticker("BTCUSDT")
        print(f"24hr change: {ticker['priceChangePercent']:.2f}%")
        print(f"24hr volume: {ticker['volume']:,.2f} BTC")
        print(f"High: ${ticker['highPrice']:,.2f}, Low: ${ticker['lowPrice']:,.2f}")
        
        # Example 6: Get order book
        print("\nGetting order book...")
        orderbook = client.get_orderbook("BTCUSDT", limit=5)
        print("Top 5 bids:")
        for price, qty in orderbook['bids'][:5]:
            print(f"  ${price:,.2f} - {qty} BTC")
        print("Top 5 asks:")
        for price, qty in orderbook['asks'][:5]:
            print(f"  ${price:,.2f} - {qty} BTC")
        
        # Example 7: Open position with valid parameters
        print("\nOpening position with valid parameters...")
        current_price = client.get_symbol_price("BTCUSDT")
        trading_info = client.get_symbol_trading_info("BTCUSDT")
        
        # Calculate order price (slightly below current for buy limit)
        order_price = client.calculate_valid_price("BTCUSDT", current_price * 0.999)
        
        # Use minimum quantity or calculate based on desired USD value
        desired_usd_value = 100  # $100 worth
        calculated_qty = desired_usd_value / order_price
        order_qty = client.calculate_valid_quantity("BTCUSDT", calculated_qty)
        
        print(f"Order price: ${order_price:,.2f}")
        print(f"Order quantity: {order_qty}")
        print(f"Order value: ${order_price * order_qty:,.2f}")
        
        # Uncomment to actually place the order
        # long_order = client.open_limit_long_position(
        #     symbol="BTCUSDT",
        #     quantity=order_qty,
        #     price=order_price,
        #     position_side="LONG"
        # )
        # print(f"Order placed: {long_order}")
        
        # Example 8: Monitor multiple symbols
        print("\nMonitoring multiple symbols...")
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        all_prices = client.get_all_prices()
        
        for symbol in symbols:
            if symbol in all_prices:
                price = all_prices[symbol]
                try:
                    min_qty = client.get_min_trade_quantity(symbol)
                    min_value = client.get_min_trade_value(symbol)
                    print(f"{symbol}: ${price:,.4f} | Min Qty: {min_qty} | Min Value: ${min_value}")
                except Exception as e:
                    print(f"{symbol}: ${price:,.4f} | Error getting limits: {e}")
        
    except Exception as e:
        print(f"Error: {e}")


def quick_price_check(symbols: list, api_key: str, api_secret: str) -> Dict[str, Dict[str, Any]]:
    """
    Quick function to check prices and trading limits for multiple symbols.
    
    Args:
        symbols: List of symbol names
        api_key: API key
        api_secret: API secret
        
    Returns:
        Dictionary with price and trading info for each symbol
    """
    client = AsterDEXClient(api_key, api_secret)
    results = {}
    
    try:
        # Get all prices at once for efficiency
        all_prices = client.get_all_prices()
        
        for symbol in symbols:
            try:
                if symbol in all_prices:
                    trading_info = client.get_symbol_trading_info(symbol)
                    results[symbol] = {
                        'price': trading_info['current_price'],
                        'min_quantity': trading_info['min_quantity'],
                        'min_value_usdt': trading_info['min_notional'],
                        'step_size': trading_info['step_size'],
                        'tick_size': trading_info['tick_size'],
                        'status': 'success'
                    }
                else:
                    results[symbol] = {'status': 'not_found'}
                    
            except Exception as e:
                results[symbol] = {'status': 'error', 'error': str(e)}
                
    except Exception as e:
        print(f"Error fetching prices: {e}")
        
    return results



if __name__ == "__main__":
    # Uncomment to run examples (make sure to add your API credentials)
    # example_usage()
    pass